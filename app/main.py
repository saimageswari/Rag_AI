from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import streamlit as st

from app.config import AppConfig, HISTORY_DIR, ensure_directories
from services.embeddings import get_embeddings
from services.exporter import export_conversation
from services.file_loader import UnsupportedFileError, load_uploaded_file
from services.llm import LLMClient
from services.models import ChatMessage, ChatTurn, LoadedDocument
from services.rag import RAGPipeline
from services.response_format import NOT_FOUND


st.set_page_config(
    page_title="AI Smart Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "documents": [],
        "uploaded_document_pool": [],
        "pipeline": None,
        "suggested_questions": [],
        "chat_messages": [],
        "pending_question": None,
        "edit_index": None,
        "last_export": None,
        "current_file_id": None,
        "current_file_chunks": [],
        "current_embeddings": None,
        "upload_signature": None,
        "suggestion_version": 0,
        "theme_mode": "Dark",
        "llm_provider": AppConfig().llm_provider,
        "groq_model": AppConfig().groq_model,
        "gemini_model": AppConfig().gemini_model,
        "openai_model": AppConfig().openai_model,
        "groq_api_key": AppConfig().groq_api_key or "",
        "google_api_key": AppConfig().google_api_key or "",
        "openai_api_key": AppConfig().openai_api_key or "",
        "ai_mode": "Zero-shot",
        "active_page": "Chat",
        "current_session_id": uuid4().hex,
        "current_session_title": "New chat",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "chat_history" in st.session_state and st.session_state.chat_history and not st.session_state.chat_messages:
        for turn in st.session_state.chat_history:
            st.session_state.chat_messages.extend(
                [
                    ChatMessage(role="user", content=turn.question, timestamp=turn.timestamp),
                    ChatMessage(
                        role="assistant",
                        content=turn.file_answer,
                        timestamp=turn.timestamp,
                        source_chunks=turn.sources,
                        ai_answer=turn.ai_answer,
                    ),
                ]
            )


def build_pipeline(documents: list[LoadedDocument], active_file_id: str) -> None:
    config = get_app_config()
    embeddings = get_embeddings(config.embedding_model)
    pipeline = RAGPipeline(config=config, embeddings=embeddings, active_file_id=active_file_id)
    pipeline.build_index(documents)
    st.session_state.pipeline = pipeline
    st.session_state.current_embeddings = embeddings
    st.session_state.current_file_chunks = pipeline.chunks
    st.session_state.suggested_questions = pipeline.suggest_questions()


def get_app_config() -> AppConfig:
    provider = st.session_state.get("llm_provider", AppConfig().llm_provider)
    return AppConfig(
        llm_provider=provider,
        groq_api_key=st.session_state.get("groq_api_key") or AppConfig().groq_api_key,
        groq_model=st.session_state.get("groq_model") or AppConfig().groq_model,
        openai_api_key=st.session_state.get("openai_api_key") or AppConfig().openai_api_key,
        openai_model=st.session_state.get("openai_model") or AppConfig().openai_model,
        google_api_key=st.session_state.get("google_api_key") or AppConfig().google_api_key,
        gemini_model=st.session_state.get("gemini_model") or AppConfig().gemini_model,
        embedding_model=AppConfig().embedding_model,
        chunk_size=AppConfig().chunk_size,
        chunk_overlap=AppConfig().chunk_overlap,
        top_k=AppConfig().top_k,
    )


def reset_active_file_state() -> None:
    st.session_state.documents = []
    st.session_state.pipeline = None
    st.session_state.suggested_questions = []
    st.session_state.chat_messages = []
    st.session_state.pending_question = None
    st.session_state.edit_index = None
    st.session_state.last_export = None
    st.session_state.current_file_chunks = []
    st.session_state.current_embeddings = None
    st.session_state.suggestion_version += 1


def reset_upload_state() -> None:
    reset_active_file_state()
    st.session_state.current_file_id = None
    st.session_state.uploaded_document_pool = []


def start_new_session(title: str = "New chat") -> None:
    st.session_state.current_session_id = uuid4().hex
    st.session_state.current_session_title = title


def set_active_documents(documents: list[LoadedDocument]) -> None:
    active_file_id = make_active_file_id(documents)
    if active_file_id == st.session_state.current_file_id:
        return
    reset_active_file_state()
    st.session_state.current_file_id = active_file_id
    st.session_state.documents = documents
    build_pipeline(documents, active_file_id)


def make_active_file_id(documents: list[LoadedDocument]) -> str:
    return "|".join(f"{doc.doc_id}:{doc.name}" for doc in documents)


def upload_signature(files) -> str:
    return "|".join(f"{file.name}:{getattr(file, 'size', len(file.getvalue()))}:{getattr(file, 'file_id', '')}" for file in files)


def answer_question(question: str) -> None:
    assistant_message = run_rag(question)
    if assistant_message:
        now = datetime.now()
        st.session_state.chat_messages.append(ChatMessage(role="user", content=question, timestamp=now))
        st.session_state.chat_messages.append(assistant_message)
        if st.session_state.current_session_title == "New chat":
            st.session_state.current_session_title = question[:48]
        save_history_session()


def run_rag(question: str, use_memory: bool = True) -> ChatMessage | None:
    pipeline: RAGPipeline | None = st.session_state.pipeline
    if not pipeline:
        with st.spinner("Generating answer..."):
            answer = LLMClient(get_app_config()).general_answer(question, conversation_history())
        return ChatMessage(
            role="assistant",
            content=answer,
            source_chunks=[],
            timestamp=datetime.now(),
        )
    retrieval_question = contextual_question(question) if use_memory else question
    with st.spinner("Retrieving context and generating a grounded answer..."):
        result = pipeline.answer(retrieval_question)
    if should_fallback_to_general(question, result.file_answer, result.sources):
        with st.spinner("Generating general answer..."):
            answer = LLMClient(get_app_config()).general_answer(question, conversation_history())
        return ChatMessage(
            role="assistant",
            content=answer,
            source_chunks=[],
            timestamp=datetime.now(),
        )
    return ChatMessage(
        role="assistant",
        content=result.file_answer,
        source_chunks=result.sources,
        timestamp=datetime.now(),
        sql_query=result.sql_query,
        excel_formula=result.excel_formula,
        csv_data=result.csv_data,
    )


def conversation_history() -> list[dict[str, str]]:
    return [
        {"role": message.role, "content": message.content}
        for message in st.session_state.chat_messages[-10:]
    ]


def should_fallback_to_general(question: str, answer: str, sources) -> bool:
    if NOT_FOUND.lower() not in answer.lower():
        return False
    if sources:
        return False
    return not looks_file_or_data_related(question)


def looks_file_or_data_related(question: str) -> bool:
    lowered = question.lower()
    data_terms = {
        "file", "document", "uploaded", "dataset", "csv", "excel", "table", "row", "column",
        "score", "marks", "salary", "sales", "revenue", "profit", "employee", "student",
        "highest", "lowest", "top", "bottom", "average", "mean", "median", "mode", "sum",
        "total", "count", "filter", "sort", "group", "duplicate", "missing", "null",
        "show data", "preview", "records", "entries",
    }
    return any(term in lowered for term in data_terms)


def contextual_question(question: str) -> str:
    if not is_followup(question):
        return question
    previous_user_questions = [
        message.content
        for message in st.session_state.chat_messages
        if message.role == "user"
    ]
    if not previous_user_questions:
        return question
    return f"Previous question: {previous_user_questions[-1]}\nFollow-up question: {question}"


def is_followup(question: str) -> bool:
    lowered = question.strip().lower()
    if len(lowered.split()) <= 4:
        return True
    return any(
        marker in lowered
        for marker in ["what about", "and ", "also", "that", "those", "them", "it", "same"]
    )


def replace_answer_pair(user_index: int, question: str) -> None:
    assistant_message = run_rag(question, use_memory=False)
    if not assistant_message:
        return
    st.session_state.chat_messages[user_index] = ChatMessage(
        role="user",
        content=question,
        timestamp=datetime.now(),
    )
    if user_index + 1 < len(st.session_state.chat_messages) and st.session_state.chat_messages[user_index + 1].role == "assistant":
        st.session_state.chat_messages[user_index + 1] = assistant_message
    else:
        st.session_state.chat_messages.insert(user_index + 1, assistant_message)


def delete_answer_pair(user_index: int) -> None:
    end = user_index + 2
    if user_index + 1 >= len(st.session_state.chat_messages) or st.session_state.chat_messages[user_index + 1].role != "assistant":
        end = user_index + 1
    del st.session_state.chat_messages[user_index:end]


def chat_turns() -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    idx = 0
    while idx < len(st.session_state.chat_messages):
        message = st.session_state.chat_messages[idx]
        if message.role == "user":
            assistant = st.session_state.chat_messages[idx + 1] if idx + 1 < len(st.session_state.chat_messages) else None
            if assistant and assistant.role == "assistant":
                turns.append(
                    ChatTurn(
                        question=message.content,
                        file_answer=assistant.content,
                        sources=assistant.source_chunks,
                        timestamp=assistant.timestamp,
                        ai_answer=assistant.ai_answer,
                        sql_query=assistant.sql_query,
                        excel_formula=assistant.excel_formula,
                        csv_data=assistant.csv_data,
                    )
                )
                idx += 2
                continue
        idx += 1
    return turns


def history_file(session_id: str) -> str:
    safe_id = "".join(ch for ch in session_id if ch.isalnum() or ch in {"_", "-"})
    return str(HISTORY_DIR / f"{safe_id}.json")


def save_history_session() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    messages = []
    for message in st.session_state.chat_messages:
        messages.append(
            {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
                "ai_answer": message.ai_answer,
                "sql_query": message.sql_query,
                "excel_formula": message.excel_formula,
                "csv_data": message.csv_data,
                "source_chunks": [asdict(source) for source in message.source_chunks],
            }
        )
    payload = {
        "id": st.session_state.current_session_id,
        "title": st.session_state.current_session_title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages,
    }
    Path(history_file(st.session_state.current_session_id)).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_history_sessions() -> list[dict]:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for path in HISTORY_DIR.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            sessions.append(
                {
                    "id": payload.get("id", path.stem),
                    "title": payload.get("title", "Untitled chat"),
                    "updated_at": payload.get("updated_at", ""),
                    "path": path,
                    "message_count": len(payload.get("messages", [])),
                }
            )
        except Exception:
            continue
    return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)


def load_history_session(session_id: str) -> None:
    path = Path(history_file(session_id))
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    reset_upload_state()
    st.session_state.upload_signature = None
    st.session_state.current_session_id = payload.get("id", session_id)
    st.session_state.current_session_title = payload.get("title", "Untitled chat")
    st.session_state.chat_messages = []
    for item in payload.get("messages", []):
        try:
            timestamp = datetime.fromisoformat(item.get("timestamp"))
        except Exception:
            timestamp = datetime.now()
        st.session_state.chat_messages.append(
            ChatMessage(
                role=item.get("role", "assistant"),
                content=item.get("content", ""),
                timestamp=timestamp,
                ai_answer=item.get("ai_answer"),
                sql_query=item.get("sql_query"),
                excel_formula=item.get("excel_formula"),
                csv_data=item.get("csv_data"),
            )
        )


def rename_history_session(session_id: str, title: str) -> None:
    path = Path(history_file(session_id))
    if not path.exists() or not title.strip():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["title"] = title.strip()
    payload["updated_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if st.session_state.current_session_id == session_id:
        st.session_state.current_session_title = title.strip()


def delete_history_session(session_id: str) -> None:
    path = Path(history_file(session_id))
    if path.exists():
        path.unlink()
    if st.session_state.current_session_id == session_id:
        reset_upload_state()
        start_new_session()


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Smart AI Assistant")
        st.caption("File intelligence, analytics, and grounded reasoning.")
        st.session_state.active_page = st.radio("Navigation", ["Chat", "Dashboard", "History", "Settings"], label_visibility="collapsed")

        st.caption(
            f"Files: {len(st.session_state.documents)} | "
            f"Chunks: {len(st.session_state.current_file_chunks)} | "
            f"Messages: {len(st.session_state.chat_messages)}"
        )

        if st.button("New chat", use_container_width=True):
            reset_upload_state()
            start_new_session()
            st.session_state.upload_signature = None
            st.rerun()

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "docx", "txt", "csv", "xlsx", "xls", "json"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            signature = upload_signature(uploaded_files)
            if signature != st.session_state.upload_signature:
                loaded_documents = []
                has_error = False
                for file in uploaded_files:
                    try:
                        loaded_documents.append(load_uploaded_file(file))
                    except UnsupportedFileError as exc:
                        st.error(str(exc))
                        has_error = True
                    except Exception as exc:
                        st.error(f"Could not process {file.name}: {exc}")
                        has_error = True
                if loaded_documents and not has_error:
                    reset_upload_state()
                    start_new_session(loaded_documents[0].name)
                    st.session_state.upload_signature = signature
                    st.session_state.uploaded_document_pool = loaded_documents
                    st.success("New upload received. Previous chat, suggestions, and vector store were reset.")

            if st.session_state.uploaded_document_pool:
                names = ["All files"] + [doc.name for doc in st.session_state.uploaded_document_pool]
                selected_name = st.selectbox("Active file", names, key="active_file_select")
                if selected_name == "All files":
                    set_active_documents(st.session_state.uploaded_document_pool)
                else:
                    selected_doc = next(doc for doc in st.session_state.uploaded_document_pool if doc.name == selected_name)
                    set_active_documents([selected_doc])
        elif st.session_state.upload_signature:
            st.session_state.upload_signature = None

        st.divider()
        st.subheader("Files")
        if st.session_state.documents:
            st.caption("Active document context only")
            for doc in st.session_state.documents:
                st.write(f"• {doc.name} ({doc.source_type}, {len(doc.text):,} chars)")
        else:
            st.caption("No documents indexed yet.")

        st.divider()
        st.subheader("Suggested Questions")
        if st.button("Regenerate suggestions", use_container_width=True, disabled=not st.session_state.pipeline):
            st.session_state.suggested_questions = st.session_state.pipeline.suggest_questions()
            st.rerun()
        if st.session_state.suggested_questions:
            for idx, question in enumerate(st.session_state.suggested_questions):
                key = f"suggested_{st.session_state.suggestion_version}_{idx}"
                if st.button(question, key=key, use_container_width=True):
                    st.session_state.pending_question = question
                    st.rerun()
        else:
            st.caption("Suggestions appear after upload.")

        st.divider()
        st.subheader("Export")
        export_filter = st.selectbox(
            "Conversation filter",
            options=["file_only", "ai_only", "full"],
            format_func=lambda x: {
                "file_only": "File Q&A only",
                "ai_only": "Explanation only",
                "full": "Full conversation",
            }[x],
        )
        export_format = st.selectbox("Format", options=["pdf", "docx", "txt"], format_func=str.upper)
        disabled = not st.session_state.chat_messages
        if st.button("Prepare export", use_container_width=True, disabled=disabled):
            path = export_conversation(chat_turns(), export_filter, export_format)
            st.session_state.last_export = path
        if st.session_state.last_export:
            path = st.session_state.last_export
            st.download_button(
                "Download export",
                data=path.read_bytes(),
                file_name=path.name,
                mime=_mime_for(path.suffix),
                use_container_width=True,
            )


def render_chat() -> None:
    render_theme()
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 1100px; }
        .source-chip {
            display: inline-block;
            padding: 0.15rem 0.45rem;
            margin: 0.1rem 0.2rem 0.1rem 0;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            background: #ffffff;
            color: #374151;
            font-size: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Smart AI Assistant")
    st.caption("Analyze files, ask questions, and get grounded answers.")

    if not st.session_state.documents:
        st.info("Upload a file to activate grounded file intelligence. Without a file, questions use the selected LLM provider.")

    start_index = max(0, len(st.session_state.chat_messages) - 10)
    if start_index:
        st.caption("Showing the last 5 turns in this session.")
    idx = start_index
    while idx < len(st.session_state.chat_messages):
        message = st.session_state.chat_messages[idx]
        if message.role != "user":
            idx += 1
            continue
        assistant = st.session_state.chat_messages[idx + 1] if idx + 1 < len(st.session_state.chat_messages) else None

        with st.chat_message("user"):
            if st.session_state.edit_index == idx:
                edited = st.text_input("Edit question", value=message.content, key=f"edit_input_{idx}")
                col_save, col_cancel = st.columns([1, 1])
                with col_save:
                    if st.button("Save and rerun", key=f"save_edit_{idx}", use_container_width=True):
                        replace_answer_pair(idx, edited)
                        st.session_state.edit_index = None
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_edit_{idx}", use_container_width=True):
                        st.session_state.edit_index = None
                        st.rerun()
            else:
                st.write(message.content)
                col_edit, col_delete, col_regen = st.columns([1, 1, 1])
                with col_edit:
                    if st.button("Edit", key=f"edit_{idx}", use_container_width=True):
                        st.session_state.edit_index = idx
                        st.rerun()
                with col_delete:
                    if st.button("Delete", key=f"delete_{idx}", use_container_width=True):
                        delete_answer_pair(idx)
                        st.session_state.edit_index = None
                        st.rerun()
                with col_regen:
                    if st.button("Regenerate", key=f"regen_{idx}", use_container_width=True):
                        replace_answer_pair(idx, message.content)
                        st.rerun()

        if not assistant:
            idx += 1
            continue
        with st.chat_message("assistant"):
            st.markdown("#### Answer")
            st.markdown(assistant.content)
            st.download_button(
                "⬇",
                data=assistant.content,
                file_name=f"response_{idx}.txt",
                mime="text/plain",
                key=f"download_response_{idx}",
            )
            with st.expander("Copy Result"):
                st.code(assistant.content, language="markdown")
            if assistant.sql_query:
                with st.expander("SQL Query"):
                    st.code(assistant.sql_query, language="sql")
            if assistant.excel_formula:
                with st.expander("Excel Formula"):
                    st.code(assistant.excel_formula, language="text")
            if assistant.csv_data:
                csv_col, excel_col = st.columns(2)
                with csv_col:
                    st.download_button(
                        "Download CSV",
                        data=assistant.csv_data,
                        file_name=f"result_{idx}.csv",
                        mime="text/csv",
                        key=f"download_csv_{idx}",
                        use_container_width=True,
                    )
                with excel_col:
                    st.download_button(
                        "Download Excel",
                        data=csv_to_xlsx_bytes(assistant.csv_data),
                        file_name=f"result_{idx}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"download_xlsx_{idx}",
                        use_container_width=True,
                    )
            if assistant.source_chunks:
                with st.expander("Sources and retrieved context"):
                    chips = []
                    for source in assistant.source_chunks:
                        page = f", page {source.page}" if source.page else ""
                        chips.append(f"<span class='source-chip'>{source.source}{page}, chunk {source.chunk_id}</span>")
                    st.markdown(" ".join(chips), unsafe_allow_html=True)
                    for source in assistant.source_chunks:
                        st.markdown(f"**{source.source}, chunk {source.chunk_id}**")
                        st.write(source.content)

            if assistant.ai_answer:
                with st.expander("Explanation", expanded=False):
                    st.write(assistant.ai_answer)
            elif st.session_state.pipeline and assistant.source_chunks:
                if st.button("Explain further", key=f"enhance_{idx}"):
                    pipeline: RAGPipeline = st.session_state.pipeline
                    context = pipeline._format_context(assistant.source_chunks)
                    with st.spinner("Preparing explanation..."):
                        assistant.ai_answer = pipeline.enhance(message.content, assistant.content, context)
                    st.rerun()
        idx += 2

    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
        answer_question(pending)
        st.rerun()

    placeholder = "Ask anything" if not st.session_state.documents else "Ask anything about your files or data"
    question = st.chat_input(placeholder)
    if question:
        answer_question(question)
        st.rerun()

    st.markdown("<div class='app-footer'>© Saivignesh — All Rights Reserved</div>", unsafe_allow_html=True)


def render_theme() -> None:
    if st.session_state.theme_mode == "Dark":
        st.markdown(
            """
            <style>
            .stApp { background: #050505; color: #f8fafc; }
            section[data-testid="stSidebar"] { background: #0f0f0f; }
            .stButton button, .stDownloadButton button { border-color: #dc2626; color: #f8fafc; background: #7f1d1d; }
            .dashboard-card {
                border: 1px solid #7f1d1d;
                background: linear-gradient(135deg, rgba(127,29,29,.38), rgba(20,20,20,.96));
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                color: #f9fafb;
                box-shadow: 0 0 24px rgba(220,38,38,.16);
            }
            .dashboard-card span { color: #fecaca; }
            .app-footer { color: #fca5a5; text-align: center; padding: 2rem 0 1rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp, .stMarkdown, .stText, p, label, span, div { color: #111827; }
            .stApp { background: #ffffff; }
            section[data-testid="stSidebar"] { background: #f9fafb; color: #111827; }
            .stButton button, .stDownloadButton button {
                border-color: #dc2626;
                color: #991b1b;
                background: #ffffff;
            }
            textarea, input, select {
                color: #111827 !important;
                background: #ffffff !important;
                border-color: #d1d5db !important;
            }
            [data-testid="stChatMessage"] {
                background: #ffffff;
                color: #111827;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
            .dashboard-card {
                border: 1px solid #fecaca;
                background: #ffffff;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                color: #111827;
                box-shadow: 0 8px 28px rgba(17,24,39,.08);
            }
            .dashboard-card span { color: #4b5563; }
            .app-footer { color: #991b1b; text-align: center; padding: 2rem 0 1rem; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def render_dashboard() -> None:
    render_theme()
    st.title("Dashboard")
    docs = st.session_state.documents
    chunks = len(st.session_state.current_file_chunks)
    messages = len(st.session_state.chat_messages)
    tables = sum(len(doc.metadata.get("tables", [])) for doc in docs)
    provider, model = active_provider_model()
    total_embeddings = chunks
    request_count = max(0, messages // 2)
    token_estimate = sum(len(message.content.split()) for message in st.session_state.chat_messages) * 2
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active files", len(docs))
    c2.metric("Indexed chunks", chunks)
    c3.metric("Messages", messages)
    c4.metric("Tables", tables)

    st.markdown(
        f"""
        <div class="dashboard-card">
            <b>Active Model</b><br>
            {model} ({provider.title()})<br>
            <span>Estimated tokens: {token_estimate:,} | Requests: {request_count}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    now = datetime.now()
    chart_rows = []
    for offset in range(6, -1, -1):
        day = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=offset)
        label = day.strftime("%b %d")
        activity = request_count if offset == 0 else 0
        chart_rows.append(
            {
                "Day": label,
                "Files": len(docs) if offset == 0 else 0,
                "Chunks": chunks if offset == 0 else 0,
                "Messages": messages if offset == 0 else 0,
                "Requests": activity,
            }
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Active files analytics")
        st.line_chart(chart_rows, x="Day", y=["Files", "Chunks"])
        st.subheader("Embedding usage")
        st.bar_chart(
            [
                {"Metric": "Embeddings", "Value": total_embeddings},
                {"Metric": "Chunks", "Value": chunks},
                {"Metric": "Tables", "Value": tables},
            ],
            x="Metric",
            y="Value",
        )
    with right:
        st.subheader("LLM API usage stats")
        st.bar_chart([{"Provider": provider, "Calls": request_count}], x="Provider", y="Calls")
        st.subheader("Message activity")
        st.line_chart(chart_rows, x="Day", y=["Messages", "Requests"])

    st.subheader("File analytics")
    if docs:
        file_rows = [
            {
                "File": doc.name,
                "Type": doc.source_type,
                "Characters": len(doc.text),
                "Tables": len(doc.metadata.get("tables", [])),
                "Status": "Indexed",
            }
            for doc in docs
        ]
        st.dataframe(file_rows, use_container_width=True, hide_index=True)
        st.progress(1.0, text="Embedding progress complete")
    else:
        st.info("Upload files to populate file analytics.")
    st.markdown("<div class='app-footer'>© Saivignesh — All Rights Reserved</div>", unsafe_allow_html=True)


def active_provider_model() -> tuple[str, str]:
    provider = st.session_state.llm_provider
    if provider == "groq":
        return provider, st.session_state.groq_model
    if provider == "gemini":
        return provider, st.session_state.gemini_model
    if provider == "openai":
        return provider, st.session_state.openai_model
    return "local", "local deterministic engine"


def csv_to_xlsx_bytes(csv_data: str) -> bytes:
    import io
    import pandas as pd

    frame = pd.read_csv(io.StringIO(csv_data))
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Result")
    return output.getvalue()


def render_settings_page() -> None:
    render_theme()
    st.title("Settings")
    previous_settings = (
        st.session_state.llm_provider,
        st.session_state.groq_model,
        st.session_state.gemini_model,
        st.session_state.openai_model,
        st.session_state.groq_api_key,
        st.session_state.google_api_key,
        st.session_state.openai_api_key,
        st.session_state.theme_mode,
        st.session_state.ai_mode,
    )
    st.session_state.theme_mode = st.radio(
        "Theme",
        ["Light", "Dark"],
        index=1 if st.session_state.theme_mode == "Dark" else 0,
        horizontal=True,
    )
    st.session_state.llm_provider = st.selectbox(
        "API Provider",
        ["local", "groq", "openai", "gemini"],
        index=["local", "groq", "openai", "gemini"].index(st.session_state.llm_provider)
        if st.session_state.llm_provider in ["local", "groq", "openai", "gemini"] else 0,
    )
    st.session_state.ai_mode = st.selectbox(
        "Prompt Mode",
        ["Zero-shot", "Few-shot", "Chain-of-thought"],
        index=["Zero-shot", "Few-shot", "Chain-of-thought"].index(st.session_state.ai_mode)
        if st.session_state.ai_mode in ["Zero-shot", "Few-shot", "Chain-of-thought"] else 0,
    )
    st.session_state.groq_model = st.text_input("Groq model", value=st.session_state.groq_model)
    update_secret("Groq API key", "groq_api_key")
    st.session_state.gemini_model = st.text_input("Gemini model", value=st.session_state.gemini_model)
    update_secret("Gemini API key", "google_api_key")
    st.session_state.openai_model = st.text_input("OpenAI model", value=st.session_state.openai_model)
    update_secret("OpenAI API key", "openai_api_key")
    st.caption("API keys are accepted as masked inputs and are not rendered back into the page after saving.")
    current_settings = (
        st.session_state.llm_provider,
        st.session_state.groq_model,
        st.session_state.gemini_model,
        st.session_state.openai_model,
        st.session_state.groq_api_key,
        st.session_state.google_api_key,
        st.session_state.openai_api_key,
        st.session_state.theme_mode,
        st.session_state.ai_mode,
    )
    if current_settings != previous_settings and st.session_state.documents:
        build_pipeline(st.session_state.documents, st.session_state.current_file_id)
    st.markdown("<div class='app-footer'>© Saivignesh — All Rights Reserved</div>", unsafe_allow_html=True)


def render_history_page() -> None:
    render_theme()
    st.title("History")
    sessions = list_history_sessions()
    search = st.text_input("Search history", placeholder="Search messages or titles")
    if search:
        lowered = search.lower()
        filtered = []
        for session in sessions:
            path = session["path"]
            text = path.read_text(encoding="utf-8").lower()
            if lowered in session["title"].lower() or lowered in text:
                filtered.append(session)
        sessions = filtered

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Clear all history", disabled=not sessions, use_container_width=True):
            for session in list_history_sessions():
                Path(session["path"]).unlink(missing_ok=True)
            reset_upload_state()
            start_new_session()
            st.rerun()
    with c2:
        st.metric("Saved sessions", len(sessions))

    if not sessions:
        st.info("No saved conversations yet.")
        return

    for session in sessions:
        with st.container(border=True):
            updated = session.get("updated_at", "")
            st.markdown(f"**{session['title']}**")
            st.caption(f"{session['message_count']} messages | {updated[:19].replace('T', ' ')}")
            new_title = st.text_input("Rename session", value=session["title"], key=f"rename_{session['id']}")
            col_open, col_rename, col_delete, col_copy = st.columns(4)
            with col_open:
                if st.button("Open", key=f"open_{session['id']}", use_container_width=True):
                    load_history_session(session["id"])
                    st.session_state.active_page = "Chat"
                    st.rerun()
            with col_rename:
                if st.button("Rename", key=f"save_rename_{session['id']}", use_container_width=True):
                    rename_history_session(session["id"], new_title)
                    st.rerun()
            with col_delete:
                if st.button("Delete", key=f"delete_history_{session['id']}", use_container_width=True):
                    delete_history_session(session["id"])
                    st.rerun()
            with col_copy:
                if st.button("Copy", key=f"copy_history_{session['id']}", use_container_width=True):
                    payload = json.loads(Path(session["path"]).read_text(encoding="utf-8"))
                    transcript = "\n\n".join(
                        f"{item.get('role', '').upper()}: {item.get('content', '')}"
                        for item in payload.get("messages", [])
                    )
                    st.code(transcript or "No messages.", language="markdown")
    st.markdown("<div class='app-footer'></div>", unsafe_allow_html=True)


def update_secret(label: str, session_key: str) -> None:
    has_key = bool(st.session_state.get(session_key))
    placeholder = "Configured. Paste a new key to replace." if has_key else "Paste API key"
    new_value = st.text_input(label, value="", placeholder=placeholder, type="password", key=f"input_{session_key}")
    if new_value:
        st.session_state[session_key] = new_value.strip()


def _mime_for(suffix: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def main() -> None:
    ensure_directories()
    init_state()
    render_sidebar()
    if st.session_state.active_page == "Dashboard":
        render_dashboard()
    elif st.session_state.active_page == "History":
        render_history_page()
    elif st.session_state.active_page == "Settings":
        render_settings_page()
    else:
        render_chat()


if __name__ == "__main__":
    main()
