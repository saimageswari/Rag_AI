from __future__ import annotations

import hashlib
import re


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def make_doc_id(name: str, text: str) -> str:
    digest = hashlib.sha256(f"{name}:{text[:5000]}:{len(text)}".encode("utf-8")).hexdigest()
    return digest[:16]


def compact_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n\n[...middle omitted for prompt length...]\n\n{tail}"
