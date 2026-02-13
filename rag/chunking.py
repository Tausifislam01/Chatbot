from __future__ import annotations

import re
from typing import Iterable, List

from rag.types import Document


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])")
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = _WS.sub(" ", text).strip()
    return text


def _split_paragraphs(text: str) -> List[str]:
    
    parts = re.split(r"\n{2,}", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) == 1:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


def _split_sentences(text: str) -> List[str]:

    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_documents(
    docs: Iterable[Document],
    *,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Document]:
    chunks: List[Document] = []

    for doc in docs:
        text = _clean(doc.text)
        paragraphs = _split_paragraphs(text)

        buffer: List[str] = []
        buffer_len = 0

        def flush():
            nonlocal buffer, buffer_len
            if not buffer:
                return
            chunk_text = " ".join(buffer).strip()
            chunks.append(
                Document(
                    text=chunk_text,
                    metadata={**doc.metadata},
                )
            )
            if overlap_chars > 0 and len(chunk_text) > overlap_chars:
                tail = chunk_text[-overlap_chars:]
                buffer = [tail]
                buffer_len = len(tail)
            else:
                buffer = []
                buffer_len = 0

        for para in paragraphs:
            para = _clean(para)
            if not para:
                continue

            units = [para] if len(para) <= target_chars else _split_sentences(para)

            for u in units:
                u = _clean(u)
                if not u:
                    continue

                if buffer_len + len(u) + 1 <= target_chars:
                    buffer.append(u)
                    buffer_len += len(u) + 1
                else:
                    flush()
                    buffer.append(u)
                    buffer_len = len(u)

        flush()

    return chunks
