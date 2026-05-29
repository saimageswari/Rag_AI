from __future__ import annotations

import re
from collections import Counter

from app.config import AppConfig
from services.response_format import NOT_FOUND, format_answer
from services.utils import compact_text

STRICT_QA_SYSTEM_PROMPT = (
    "You are a strict document-based AI assistant with file memory. "
    "You must answer ONLY using the provided context. Do NOT use external knowledge. "
    "If answer is not present, say exactly 'The requested information is not available in the provided file.'. "
    "Every answer must have two sections: FINAL RESULT and EXPLANATION."
)

STRICT_QA_RULES = """Rules:
- Be precise.
- If the question asks for a number, calculate ONLY from context.
- If the question asks for a list, extract EXACT values from context.
- If the question asks for an explanation, summarize ONLY the context.
- Do NOT hallucinate.
- Do NOT mention facts that are not in the context.
- Format exactly with:
  FINAL RESULT
  EXPLANATION
"""

STRICT_EXPLANATION_SYSTEM_PROMPT = (
    "Explain the following answer in simple terms using ONLY given document context. "
    "Add examples ONLY if present in context. Do NOT use general knowledge."
)

STRICT_SUGGESTION_SYSTEM_PROMPT = (
    "Generate 10 highly relevant questions ONLY from this provided file content. "
    "Do NOT use external knowledge. Each question must be grounded in actual topics, "
    "names, projects, datasets, numbers, or sections from the document."
)


class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

    @property
    def provider_name(self) -> str:
        provider = self.config.llm_provider
        if provider == "groq" and self.config.groq_api_key:
            return "groq"
        if provider == "gemini" and self.config.google_api_key:
            return "gemini"
        if provider == "openai" and self.config.openai_api_key:
            return "openai"
        return "local"

    def grounded_answer(self, question: str, context: str) -> str:
        if not context.strip():
            return format_answer(NOT_FOUND)
        structured = self._structured_answer(question, context)
        if structured:
            return self._ensure_formatted(structured)
        if self.provider_name == "local":
            return self._ensure_formatted(self._local_grounded_answer(question, context))
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{compact_text(context, 16000)}\n\n"
            f"{STRICT_QA_RULES}\n"
            "Answer:"
        )
        answer = self._generate(
            system_prompt=STRICT_QA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            fallback_question=question,
            fallback_context=context,
            task="answer",
        )
        return self._ensure_formatted(self._clean_model_answer(answer))

    def ai_explanation(self, question: str, file_answer: str, context: str) -> str:
        if not file_answer.strip() or "not found" in file_answer.lower():
            return NOT_FOUND
        user_prompt = (
            f"Question: {question}\n\n"
            f"Grounded file answer:\n{file_answer}\n\n"
            f"Retrieved context:\n{compact_text(context, 10000)}\n\n"
            "Explain the answer simply using only the retrieved context:"
        )
        if self.provider_name == "local":
            return self._local_explanation(file_answer)
        return self._clean_model_answer(
            self._generate(
                system_prompt=STRICT_EXPLANATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                fallback_question=question,
                fallback_context=context,
                task="explain",
            )
        )

    def general_answer(self, question: str, history: list[dict[str, str]] | None = None) -> str:
        system_prompt = (
            "You are a helpful, conversational AI assistant. Answer directly and professionally. "
            "If the user provided a paragraph in the question, analyze that text. "
            "Support coding, math, reasoning, writing, debugging, and normal conversation. "
            "Do not claim file context unless a file was provided."
        )
        if self.provider_name == "local":
            return (
                "FINAL RESULT\n"
                "I can answer file and table questions locally. For normal ChatGPT-like conversation, configure Groq, OpenAI, or Gemini in Settings.\n\n"
                "EXPLANATION\n"
                "Local mode is optimized for grounded file extraction and deterministic table calculations."
            )
        if history:
            compact_history = "\n".join(
                f"{item['role']}: {compact_text(item['content'], 1200)}"
                for item in history[-8:]
            )
            user_prompt = f"Conversation so far:\n{compact_history}\n\nUser message:\n{question}"
        else:
            user_prompt = question
        return self._generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            fallback_question=question,
            fallback_context="",
            task="general",
        )

    def suggested_questions(self, document_text: str) -> list[str]:
        local_questions = self._local_suggested_questions(document_text)
        if self.provider_name == "local":
            return local_questions
        user_prompt = (
            "Return one question per line. Do not number the questions.\n\n"
            f"Document:\n{compact_text(document_text, 14000)}"
        )
        raw = self._generate(
            system_prompt=STRICT_SUGGESTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
            fallback_question="",
            fallback_context=document_text,
            task="suggest",
        )
        questions = [line.strip(" -0123456789.\t") for line in raw.splitlines() if line.strip()]
        questions = [q if q.endswith("?") else f"{q}?" for q in questions if len(q.split()) >= 4]
        merged: list[str] = []
        for question in questions + local_questions:
            if question not in merged:
                merged.append(question)
        return merged[:10]

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        fallback_question: str,
        fallback_context: str,
        task: str,
    ) -> str:
        try:
            if self.provider_name == "groq":
                return self._generate_groq(system_prompt, user_prompt, temperature)
            if self.provider_name == "gemini":
                return self._generate_gemini(system_prompt, user_prompt, temperature)
            if self.provider_name == "openai":
                return self._generate_openai(system_prompt, user_prompt, temperature)
        except Exception:
            if task == "suggest":
                return "\n".join(self._local_suggested_questions(fallback_context))
            if task == "explain":
                answer = self._extract_prompt_part(user_prompt, "Grounded file answer:\n", "\n\nRetrieved context:")
                return self._local_explanation(answer)
            if task == "general":
                return (
                    "FINAL RESULT\n"
                    "The selected API provider could not generate a response.\n\n"
                    "EXPLANATION\n"
                    "Check the API key, model name, and internet connection. Secrets are not shown for security."
                )
            return self._local_grounded_answer(fallback_question, fallback_context)
        return self._local_grounded_answer(fallback_question, fallback_context)

    def _generate_groq(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        from groq import Groq

        client = Groq(api_key=self.config.groq_api_key)
        response = client.chat.completions.create(
            model=self.config.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _generate_gemini(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.config.google_api_key)
        model = genai.GenerativeModel(
            self.config.gemini_model,
            system_instruction=system_prompt,
            generation_config={"temperature": temperature},
        )
        response = model.generate_content(user_prompt)
        return (response.text or "").strip()

    def _generate_openai(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.config.openai_api_key)
        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _local_grounded_answer(self, question: str, context: str) -> str:
        clean_context = self._strip_source_labels(context)
        if not clean_context:
            return NOT_FOUND

        structured = self._structured_answer(question, context)
        if structured:
            return structured

        question_lower = question.lower()
        intent = self._detect_intent(question_lower)
        if intent:
            section_answer = self._answer_from_sections(intent, clean_context)
            if section_answer:
                return section_answer

        dataset_answer = self._answer_dataset_question(question_lower, clean_context)
        if dataset_answer:
            return dataset_answer

        relevant = self._relevant_lines(question, clean_context)
        if relevant:
            if self._is_count_question(question_lower):
                return self._count_answer(question_lower, relevant)
            return "\n".join(f"- {line}" for line in relevant[:6])

        return NOT_FOUND

    def _structured_answer(self, question: str, context: str) -> str | None:
        clean_context = self._strip_source_labels(context)
        question_lower = question.lower()
        if not clean_context:
            return NOT_FOUND

        if self._is_count_question(question_lower):
            count_target = self._count_target(question_lower)
            if count_target == "dataset":
                dataset_answer = self._answer_dataset_question(question_lower, clean_context)
                return dataset_answer or NOT_FOUND
            if count_target == "project":
                projects = self._project_entries_from_lines(self._project_context_lines(clean_context))
                if projects:
                    names = [project["title"] for project in projects]
                    raw = f"Count of projects = {len(names)}\nProjects: " + "; ".join(names)
                    return format_answer(raw, f"The provided file lists {len(names)} project(s): {', '.join(names)}.")
                return NOT_FOUND
            if count_target == "name":
                names = self._extract_people(clean_context)
                if names:
                    raw = f"Count of names = {len(names)}\nNames: " + "; ".join(names)
                    return format_answer(raw, f"The provided file contains {len(names)} extracted name(s): {', '.join(names)}.")
                return NOT_FOUND
            if count_target == "skill":
                skills = self._skills_from_context(clean_context)
                if skills:
                    raw = f"Count of skills/technologies = {len(skills)}\nSkills/technologies: " + "; ".join(skills)
                    return format_answer(raw, f"The provided file lists {len(skills)} skill or technology item(s).")
                return NOT_FOUND

        if self._is_list_question(question_lower):
            intent = self._detect_intent(question_lower)
            if intent:
                answer = self._answer_from_sections(intent, clean_context)
                if answer:
                    return answer
            if "name" in question_lower:
                names = self._extract_people(clean_context)
                if names:
                    raw = "Names found in the provided file:\n" + "\n".join(f"- {name}" for name in names)
                    return format_answer(raw, "These are the exact name entries extracted from the provided file.")

        return None

    @staticmethod
    def _local_explanation(file_answer: str) -> str:
        lines = [line.strip(" -") for line in file_answer.splitlines() if line.strip()]
        if not lines:
            return NOT_FOUND
        if len(lines) == 1:
            return f"In simple terms: {lines[0]}"
        return "In simple terms:\n" + "\n".join(f"- {line}" for line in lines[:8])

    def _local_suggested_questions(self, document_text: str) -> list[str]:
        sections = self._extract_sections(document_text)
        topics = self._extract_topics(document_text, sections)
        questions: list[str] = []

        for project in topics["projects"][:4]:
            questions.append(f"What does the document say about {project}?")
        for dataset in topics["datasets"][:3]:
            questions.append(f"What information is provided about {dataset}?")
        for person in topics["people"][:3]:
            questions.append(f"What role or context is provided for {person}?")
        for heading, _ in sections[:8]:
            if self._is_useful_question_heading(heading):
                questions.append(f"What information is in the {heading} section?")
        for skill in topics["skills"][:4]:
            questions.append(f"How is {skill} mentioned in the document?")

        if topics["datasets"]:
            questions.append("How many datasets are mentioned in the document?")
        if topics["projects"]:
            questions.append("Which projects are listed in the document?")
        if topics["skills"]:
            questions.append("Which skills or technologies are listed in the document?")

        cleaned: list[str] = []
        for question in questions:
            question = re.sub(r"\s+", " ", question).strip()
            if len(question.split()) >= 5 and question not in cleaned:
                cleaned.append(question)
        return cleaned[:10] or self._fallback_questions_from_keywords(document_text)

    def _answer_dataset_question(self, question_lower: str, context: str) -> str | None:
        dataset_matches = re.findall(
            r"\bDataset\s+([A-Za-z0-9_-]+)\s*:\s*([^\n\r]+)",
            context,
            flags=re.IGNORECASE,
        )
        if not dataset_matches or "dataset" not in question_lower:
            return None

        if self._is_count_question(question_lower):
            names = [f"Dataset {code}: {name.strip(' -')}" for code, name in dataset_matches]
            return f"There are {len(dataset_matches)} datasets mentioned: " + "; ".join(names) + "."

        records = re.findall(
            r"(Dataset\s+[A-Za-z0-9_-]+:\s*[^\n\r]+).*?Total records:\s*([0-9,]+)",
            context,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if records:
            return "Datasets found in the retrieved context:\n" + "\n".join(
                f"- {name.strip()} - {count} total records" for name, count in records
            )
        return "Datasets found in the retrieved context:\n" + "\n".join(
            f"- Dataset {code}: {name.strip(' -')}" for code, name in dataset_matches
        )

    def _project_context_lines(self, context: str) -> list[str]:
        sections = self._extract_sections(context)
        lines: list[str] = []
        for heading, section_lines in sections:
            if "project" in heading.lower():
                lines.extend(section_lines)
        return lines or self._extract_project_like_blocks(context)

    def _skills_from_context(self, context: str) -> list[str]:
        sections = self._extract_sections(context)
        lines: list[str] = []
        for heading, section_lines in sections:
            if any(word in heading.lower() for word in ["skill", "technolog", "tool", "language"]):
                lines.extend(section_lines)
        return self._dedupe(self._split_skill_lines(lines))

    def _extract_people(self, context: str) -> list[str]:
        names = []
        for line in context.splitlines():
            lower = line.lower()
            if any(skip in lower for skip in ["dataset", "project", "skills", "technologies", "workflow", "names mentioned"]):
                continue
            if "(" in line:
                candidate = line.split("(", 1)[0].strip(" -•")
                if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}", candidate):
                    names.append(candidate)
                continue
            names.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", line))
        return self._dedupe([
            name for name in names
            if name.lower() not in {"total records", "pasted text", "file grounded", "retrieved context", "provided file"}
            and not any(role in name.lower() for role in ["intern", "scientist", "engineer", "lead", "manager", "developer"])
        ])

    @staticmethod
    def _strip_source_labels(context: str) -> str:
        context = re.sub(r"\[source:[^\]]+\]", "", context)
        context = re.sub(r"\r\n?", "\n", context)
        context = re.sub(r"\n{3,}", "\n\n", context)
        return context.strip()

    @staticmethod
    def _extract_prompt_part(prompt: str, start: str, end: str) -> str:
        if start not in prompt:
            return ""
        value = prompt.split(start, 1)[1]
        if end in value:
            value = value.split(end, 1)[0]
        return value.strip()

    @staticmethod
    def _detect_intent(question_lower: str) -> str | None:
        intent_words = {
            "projects": ["project", "projects", "portfolio"],
            "skills": ["skill", "skills", "technology", "technologies", "tools", "language", "languages"],
            "experience": ["experience", "work", "internship", "job", "employment"],
            "education": ["education", "degree", "college", "university", "school", "cgpa", "gpa"],
            "certifications": ["certificate", "certification", "certifications", "course", "courses"],
            "contact": ["email", "phone", "mobile", "linkedin", "github", "contact"],
            "summary": ["summary", "summarize", "profile", "objective", "about"],
            "achievements": ["achievement", "achievements", "award", "awards"],
        }
        for intent, words in intent_words.items():
            if any(word in question_lower for word in words):
                return intent
        return None

    def _answer_from_sections(self, intent: str, text: str) -> str | None:
        if intent == "contact":
            return self._extract_contact(text)

        sections = self._extract_sections(text)
        aliases = {
            "projects": ["project", "projects", "academic projects", "personal projects"],
            "skills": ["skill", "skills", "technical skills", "technologies", "tools", "languages"],
            "experience": ["experience", "work experience", "internship", "employment"],
            "education": ["education", "academic", "qualification"],
            "certifications": ["certification", "certifications", "courses", "training"],
            "summary": ["summary", "profile", "objective", "about"],
            "achievements": ["achievement", "achievements", "awards"],
        }
        wanted = aliases.get(intent, [intent])
        matched: list[str] = []
        for heading, lines in sections:
            heading_lower = heading.lower()
            if any(alias in heading_lower for alias in wanted):
                matched.extend(lines)

        if intent == "projects":
            project_answer = self._format_project_answer(matched or self._extract_project_like_blocks(text))
            if project_answer:
                return project_answer
        if not matched:
            return None

        cleaned = self._clean_answer_lines(matched)
        if not cleaned:
            return None

        title = {
            "projects": "Projects found in the retrieved context:",
            "skills": "Skills found in the retrieved context:",
            "experience": "Experience found in the retrieved context:",
            "education": "Education found in the retrieved context:",
            "certifications": "Certifications/courses found in the retrieved context:",
            "summary": "Summary/profile found in the retrieved context:",
            "achievements": "Achievements found in the retrieved context:",
        }.get(intent, "Information found in the retrieved context:")
        return title + "\n" + "\n".join(f"- {line}" for line in cleaned[:14])

    def _format_project_answer(self, lines: list[str]) -> str | None:
        projects = self._project_entries_from_lines(lines)
        if not projects:
            return None
        answer_lines = []
        for project in projects[:8]:
            title = project["title"]
            detail = project.get("detail")
            if detail:
                answer_lines.append(f"- {title}: {detail}")
            else:
                answer_lines.append(f"- {title}")
        return "Projects found in the retrieved context:\n" + "\n".join(answer_lines)

    @staticmethod
    def _extract_sections(text: str) -> list[tuple[str, list[str]]]:
        known = {
            "summary", "profile", "objective", "about", "education", "academic", "qualification",
            "skills", "technical skills", "technologies", "tools", "experience", "work experience",
            "internship", "employment", "projects", "project", "academic projects", "personal projects",
            "certifications", "certification", "courses", "training", "achievements", "awards",
            "publications", "languages", "interests", "dataset information", "important names mentioned",
            "technologies used", "workflow",
        }
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections: list[tuple[str, list[str]]] = []
        current_heading = "Document"
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_lines
            if current_lines:
                sections.append((current_heading, current_lines))
                current_lines = []

        for line in lines:
            normalized = re.sub(r"^[#\-\d.\s]+", "", line)
            normalized = re.sub(r"[:|•\-]+$", "", normalized).strip()
            lower = normalized.lower()
            alpha = re.sub(r"[^A-Za-z ]", "", normalized).strip()
            upper_ratio = sum(1 for c in alpha if c.isupper()) / max(1, len(alpha.replace(" ", "")))
            is_short = 2 <= len(normalized) <= 60
            is_heading = is_short and (
                lower in known
                or any(k == lower for k in known)
                or (upper_ratio > 0.75 and len(alpha.split()) <= 5)
            )
            if is_heading:
                flush()
                current_heading = normalized
            else:
                current_lines.append(line)
        flush()
        return sections

    @staticmethod
    def _clean_answer_lines(lines: list[str]) -> list[str]:
        cleaned: list[str] = []
        for line in lines:
            line = re.sub(r"^[•*\-–\d.)\s]+", "", line).strip()
            line = re.sub(r"\s{2,}", " ", line)
            if not line or len(line) < 2:
                continue
            if line.lower().startswith(("source:", "page ")):
                continue
            if line not in cleaned:
                cleaned.append(line)
        return cleaned

    def _extract_project_like_blocks(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        candidates = [
            line
            for line in lines
            if any(
                word in line.lower()
                for word in ["project", "app", "system", "assistant", "prediction", "detection", "platform", "analyzer"]
            )
        ]
        return candidates

    @staticmethod
    def _extract_contact(text: str) -> str | None:
        emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
        phones = re.findall(r"(?:\+?\d[\d\s().-]{7,}\d)", text)
        links = re.findall(r"https?://\S+|(?:linkedin|github)\.com/\S+", text, flags=re.IGNORECASE)
        lines = []
        if emails:
            lines.append("Email: " + ", ".join(dict.fromkeys(emails)))
        if phones:
            lines.append("Phone: " + ", ".join(dict.fromkeys(phone.strip() for phone in phones[:3])))
        if links:
            lines.append("Links: " + ", ".join(dict.fromkeys(links)))
        if not lines:
            return None
        return "\n".join(f"- {line}" for line in lines)

    def _extract_topics(self, text: str, sections: list[tuple[str, list[str]]]) -> dict[str, list[str]]:
        projects = []
        skills = []
        for heading, lines in sections:
            lower = heading.lower()
            if "project" in lower:
                projects.extend(project["title"] for project in self._project_entries_from_lines(lines))
            if any(word in lower for word in ["skill", "technolog", "tool", "language"]):
                skills.extend(self._split_skill_lines(lines))

        datasets = [
            f"Dataset {code}: {name.strip(' -')}"
            for code, name in re.findall(r"\bDataset\s+([A-Za-z0-9_-]+)\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
        ]
        people = []
        for line in text.splitlines():
            if "(" in line and ")" in line:
                people.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", line))
        people = [
            p for p in people
            if p.lower() not in {"total records", "dataset information"}
            and not any(word in p.lower() for word in ["assistant", "analyzer", "detection", "resume", "study"])
        ]

        return {
            "projects": self._dedupe(projects),
            "skills": self._dedupe(skills),
            "datasets": self._dedupe(datasets),
            "people": self._dedupe(people),
        }

    def _project_entries_from_lines(self, lines: list[str]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        current: dict[str, str] | None = None
        for line in lines:
            original = line.strip()
            line = re.sub(r"^[•*\-–\d.)\s]+", "", original).strip()
            if not line:
                continue
            if self._looks_like_project_title(line):
                if current:
                    entries.append(current)
                current = {"title": self._project_title(line), "detail": ""}
                continue
            if current and self._looks_like_project_detail(line):
                detail = current.get("detail", "")
                current["detail"] = (detail + " " + line).strip() if detail else line
        if current:
            entries.append(current)
        if entries:
            return entries

        fallback = []
        for line in lines:
            cleaned = re.sub(r"^[•*\-–\d.)\s]+", "", line).strip()
            if self._looks_like_project_title(cleaned):
                fallback.append({"title": self._project_title(cleaned), "detail": ""})
        return fallback

    @staticmethod
    def _looks_like_project_title(line: str) -> bool:
        lower = line.lower()
        if len(line) > 140:
            return False
        if lower.startswith(("built ", "created ", "developed ", "designed ", "architected ", "implemented ", "used ", "improved ", "deployed ", "achieved ", "containerized ", "outperforming ")):
            return False
        title_signals = ["|", "project", "assistant", "chatbot", "agent", "platform", "system", "app", "analyzer", "recommendation", "analysis", "detection", "tracker"]
        return any(signal in lower for signal in title_signals)

    @staticmethod
    def _looks_like_project_detail(line: str) -> bool:
        lower = line.lower()
        return len(line) > 15 and (
            lower.startswith(("built ", "created ", "developed ", "designed ", "architected ", "implemented ", "used ", "improved ", "deployed ", "achieved ", "containerized "))
            or any(word in lower for word in ["achieving", "accuracy", "retrieval", "baseline", "api", "model", "data", "users"])
        )

    @staticmethod
    def _project_title(line: str) -> str:
        title = re.split(r"\s{2,}|\s+[|]\s+", line, maxsplit=1)[0].strip()
        title = re.sub(r"\s+", " ", title).strip(" -:|")
        return title or line.strip()

    @staticmethod
    def _split_skill_lines(lines: list[str]) -> list[str]:
        skills = []
        for line in lines:
            line = re.sub(r"^[•*\-–\d.)\s]+", "", line)
            pieces = re.split(r"[,|/;]", line)
            skills.extend(piece.strip() for piece in pieces if 2 <= len(piece.strip()) <= 40)
        return skills

    @staticmethod
    def _fallback_questions_from_keywords(text: str) -> list[str]:
        words = [
            word
            for word in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", text)
            if word.lower() not in {"the", "and", "for", "with", "from", "this", "that", "document"}
        ]
        common = [word for word, _ in Counter(words).most_common(10)]
        return [f"What does the document say about {word}?" for word in common[:10]]

    @staticmethod
    def _is_useful_question_heading(heading: str) -> bool:
        lower = heading.lower()
        useful = {
            "summary", "profile", "objective", "education", "skills", "technical skills",
            "experience", "work experience", "internship", "projects", "certifications",
            "courses", "achievements", "dataset information", "important names mentioned",
            "technologies used", "workflow",
        }
        return lower in useful

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen = []
        for value in values:
            value = re.sub(r"\s+", " ", value).strip(" -:")
            if value and value not in seen:
                seen.append(value)
        return seen

    @staticmethod
    def _is_count_question(question_lower: str) -> bool:
        return any(term in question_lower for term in ["how many", "number of", "count", "total"])

    @staticmethod
    def _is_list_question(question_lower: str) -> bool:
        return any(term in question_lower for term in ["list", "show", "tell me", "what are", "which are", "give me"])

    @staticmethod
    def _count_target(question_lower: str) -> str | None:
        targets = {
            "dataset": ["dataset", "datasets"],
            "project": ["project", "projects"],
            "name": ["name", "names", "people", "person"],
            "skill": ["skill", "skills", "technology", "technologies", "tool", "tools"],
        }
        for target, words in targets.items():
            if any(word in question_lower for word in words):
                return target
        return None

    @staticmethod
    def _clean_model_answer(answer: str) -> str:
        answer = answer.strip()
        if not answer:
            return NOT_FOUND
        lowered = answer.lower()
        if "not found" in lowered or "not present" in lowered or "not in the context" in lowered:
            return NOT_FOUND
        return answer

    @staticmethod
    def _ensure_formatted(answer: str) -> str:
        if not answer or answer.strip() == NOT_FOUND:
            return format_answer(NOT_FOUND)
        if "FINAL RESULT" in answer and "EXPLANATION" in answer:
            return answer.strip()
        return format_answer(answer)

    @staticmethod
    def _count_answer(question_lower: str, lines: list[str]) -> str:
        noun = "items"
        for candidate in ["project", "skill", "dataset", "certificate", "course", "experience"]:
            if candidate in question_lower:
                noun = candidate + ("s" if not candidate.endswith("s") else "")
                break
        return f"There are {len(lines)} {noun} found in the retrieved context:\n" + "\n".join(
            f"- {line}" for line in lines[:10]
        )

    @staticmethod
    def _relevant_lines(question: str, context: str) -> list[str]:
        stopwords = {
            "what", "which", "who", "when", "where", "why", "how", "many", "much", "the",
            "is", "are", "was", "were", "do", "does", "did", "in", "of", "to", "from",
            "about", "tell", "list", "show", "give", "me", "any", "document", "uploaded",
        }
        keywords = {
            token
            for token in re.findall(r"[a-zA-Z0-9_+#.-]+", question.lower())
            if token not in stopwords and len(token) > 2
        }
        if not keywords:
            return []
        lines = [line.strip(" -") for line in re.split(r"[\n\r]+", context) if line.strip(" -")]
        scored = []
        for line in lines:
            lower = line.lower()
            score = sum(1 for keyword in keywords if keyword in lower)
            if score:
                scored.append((score, len(line), line))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return self_dedupe([line for _, _, line in scored])


def self_dedupe(values: list[str]) -> list[str]:
    seen = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen
