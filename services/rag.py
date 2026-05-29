from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import AppConfig
from services.llm import LLMClient
from services.models import LoadedDocument, SourceChunk
from services.response_format import NOT_FOUND, format_answer
from services.table_qa import TableQA


@dataclass
class RAGAnswer:
    file_answer: str
    context: str
    sources: list[SourceChunk]
    sql_query: str | None = None
    excel_formula: str | None = None
    csv_data: str | None = None


class RAGPipeline:
    def __init__(self, config: AppConfig, embeddings, active_file_id: str):
        self.config = config
        self.embeddings = embeddings
        self.active_file_id = active_file_id
        self.llm = LLMClient(config)
        self.vectorstore: FAISS | None = None
        self.documents: list[LoadedDocument] = []
        self.chunks: list[Document] = []
        self.table_qa = TableQA([])

    def build_index(self, documents: list[LoadedDocument]) -> None:
        self.documents = documents
        self.table_qa = TableQA(documents)
        self.chunks = self._chunk_documents(documents)
        if not self.chunks:
            self.vectorstore = None
            return
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)

    def answer(self, question: str) -> RAGAnswer:
        table_answer = self.table_qa.answer(question)
        if table_answer:
            artifacts = self.table_qa.artifacts(question)
            return RAGAnswer(
                file_answer=table_answer,
                context="",
                sources=[],
                sql_query=artifacts.get("sql_query"),
                excel_formula=artifacts.get("excel_formula"),
                csv_data=artifacts.get("csv_data"),
            )
        if self.vectorstore is None:
            return RAGAnswer(
                file_answer="Upload a document or paste text before asking questions.",
                context="",
                sources=[],
            )
        sources = self._retrieve(question, top_k=min(self.config.top_k, 5))
        if not sources:
            return RAGAnswer(
                file_answer=format_answer(NOT_FOUND),
                context="",
                sources=[],
            )
        context = self._format_context(sources)
        answer = self.llm.grounded_answer(question, context)
        return RAGAnswer(file_answer=answer, context=context, sources=sources)

    def enhance(self, question: str, file_answer: str, context: str) -> str:
        return self.llm.ai_explanation(question, file_answer, context)

    def suggest_questions(self) -> list[str]:
        combined = "\n\n".join(doc.text for doc in self.documents)
        questions = []
        questions.extend(self.table_qa.suggestions())
        questions.extend(self.llm.suggested_questions(combined))
        deduped = []
        for question in questions:
            if question not in deduped:
                deduped.append(question)
        return deduped[:10]

    def _retrieve(self, question: str, top_k: int) -> list[SourceChunk]:
        candidates: dict[tuple[str, int], tuple[Document, float]] = {}

        vector_hits = self.vectorstore.similarity_search_with_score(question, k=max(top_k * 3, 8))
        vector_scores = [float(score) for _, score in vector_hits]
        max_score = max(vector_scores) if vector_scores else 1.0
        min_score = min(vector_scores) if vector_scores else 0.0
        span = max(max_score - min_score, 1e-6)
        for doc, score in vector_hits:
            if doc.metadata.get("active_file_id") != self.active_file_id:
                continue
            key = (doc.metadata.get("doc_id", ""), int(doc.metadata.get("chunk_id", 0)))
            normalized = 1.0 - ((float(score) - min_score) / span)
            candidates[key] = (doc, max(candidates.get(key, (doc, 0.0))[1], normalized))

        query_terms = self._query_terms(question)
        expanded_terms = self._expanded_query_terms(question, query_terms)
        for doc in self.chunks:
            if doc.metadata.get("active_file_id") != self.active_file_id:
                continue
            keyword_score = self._keyword_score(doc.page_content, expanded_terms)
            if keyword_score <= 0:
                continue
            key = (doc.metadata.get("doc_id", ""), int(doc.metadata.get("chunk_id", 0)))
            existing = candidates.get(key, (doc, 0.0))[1]
            candidates[key] = (doc, max(existing, min(keyword_score, 1.5)))

        ranked = sorted(candidates.values(), key=lambda item: item[1], reverse=True)
        filtered = [(doc, score) for doc, score in ranked if score >= 0.08]
        selected = filtered[:top_k] or ranked[:top_k]

        return [
            SourceChunk(
                source=doc.metadata.get("source", "document"),
                chunk_id=int(doc.metadata.get("chunk_id", idx + 1)),
                doc_id=doc.metadata.get("doc_id"),
                page=doc.metadata.get("page"),
                content=doc.page_content,
                score=float(score),
            )
            for idx, (doc, score) in enumerate(selected)
        ]

    def _chunk_documents(self, documents: list[LoadedDocument]) -> list[Document]:
        splitter = self._get_splitter()
        chunks: list[Document] = []
        for loaded in documents:
            page_docs = self._page_documents(loaded)
            split_docs = splitter.split_documents(page_docs)
            for idx, doc in enumerate(split_docs, start=1):
                doc.metadata.update(
                    {
                        "source": loaded.name,
                        "doc_id": loaded.doc_id,
                        "active_file_id": self.active_file_id,
                        "chunk_id": idx,
                    }
                )
                chunks.append(doc)
        return chunks

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        try:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=max(self.config.chunk_size, 650),
                chunk_overlap=max(self.config.chunk_overlap, 80),
            )
        except ImportError:
            return RecursiveCharacterTextSplitter(
                chunk_size=max(self.config.chunk_size, 650) * 4,
                chunk_overlap=max(self.config.chunk_overlap, 80) * 4,
            )

    @staticmethod
    def _query_terms(question: str) -> set[str]:
        stopwords = {
            "what", "which", "who", "when", "where", "why", "how", "many", "much", "the",
            "is", "are", "was", "were", "do", "does", "did", "in", "of", "to", "from",
            "about", "tell", "list", "show", "give", "me", "any", "document", "uploaded",
        }
        return {
            token
            for token in re.findall(r"[a-zA-Z0-9_+#.-]+", question.lower())
            if token not in stopwords and len(token) > 2
        }

    @staticmethod
    def _expanded_query_terms(question: str, terms: set[str]) -> set[str]:
        lower = question.lower()
        expanded = set(terms)
        if "project" in lower:
            expanded.update({"project", "projects", "app", "system", "assistant", "platform", "analyzer"})
        if "skill" in lower or "technology" in lower or "tool" in lower:
            expanded.update({"skills", "technical", "technologies", "tools", "python", "java", "sql"})
        if "experience" in lower or "intern" in lower or "work" in lower:
            expanded.update({"experience", "internship", "work", "company", "role"})
        if "education" in lower or "degree" in lower:
            expanded.update({"education", "degree", "college", "university", "cgpa", "gpa"})
        if "dataset" in lower:
            expanded.update({"dataset", "records", "columns"})
        return expanded

    @staticmethod
    def _keyword_score(text: str, terms: set[str]) -> float:
        if not terms:
            return 0.0
        lower = text.lower()
        hits = sum(1 for term in terms if term in lower)
        heading_bonus = 0.4 if any(re.search(rf"(?im)^\s*{re.escape(term)}s?\s*:?\s*$", lower) for term in terms) else 0.0
        return (hits / max(len(terms), 1)) + heading_bonus

    @staticmethod
    def _page_documents(loaded: LoadedDocument) -> list[Document]:
        if "[Page " not in loaded.text:
            return [Document(page_content=loaded.text, metadata={"source": loaded.name})]
        docs = []
        sections = loaded.text.split("[Page ")
        for section in sections:
            if not section.strip():
                continue
            page_label, _, content = section.partition("]")
            try:
                page = int(page_label.strip())
            except ValueError:
                page = None
            docs.append(Document(page_content=content.strip(), metadata={"source": loaded.name, "page": page}))
        return docs or [Document(page_content=loaded.text, metadata={"source": loaded.name})]

    @staticmethod
    def _format_context(sources: list[SourceChunk]) -> str:
        formatted = []
        for source in sources:
            page = f", page {source.page}" if source.page else ""
            formatted.append(
                f"[source: {source.source}{page}, chunk {source.chunk_id}]\n{source.content}"
            )
        return "\n\n".join(formatted)
