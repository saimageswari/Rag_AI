from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class LoadedDocument:
    doc_id: str
    name: str
    source_type: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SourceChunk:
    source: str
    chunk_id: int
    content: str
    doc_id: str | None = None
    page: int | None = None
    score: float | None = None


@dataclass
class ChatTurn:
    question: str
    file_answer: str
    sources: list[SourceChunk]
    timestamp: datetime
    ai_answer: str | None = None
    sql_query: str | None = None
    excel_formula: str | None = None
    csv_data: str | None = None


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    source_chunks: list[SourceChunk] = field(default_factory=list)
    ai_answer: str | None = None
    sql_query: str | None = None
    excel_formula: str | None = None
    csv_data: str | None = None


ExportFilter = Literal["file_only", "ai_only", "full"]
