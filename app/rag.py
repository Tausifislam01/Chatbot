from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from groq import Groq
from sentence_transformers import SentenceTransformer

from app.settings import Settings
from rag.embed_store import load_store
from rag.prompts import SYSTEM_PROMPT
from rag.retrieve import retrieve
from rag.types import Document


@dataclass(frozen=True)
class Answer:
    text: str
    used_chunks: List[Tuple[Document, float]]
    rejected: bool


def _format_context(chunks: List[Tuple[Document, float]], max_chars: int) -> str:
    parts: List[str] = []
    total = 0
    for doc, score in chunks:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "n/a")
        block = f"[source: {src}, page: {page}, score: {score:.3f}]\n{doc.text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def answer_question(settings: Settings, question: str) -> Answer:
    store_dir = Path(settings.DATA_DIR) / "processed" / settings.COMPANY_ID
    index, chunks = load_store(store_dir)

    embed_model = SentenceTransformer(settings.EMBED_MODEL)

    retrieved = retrieve(
        embed_model=embed_model,
        faiss_index=index,
        chunks=chunks,
        query=question,
        top_k=settings.TOP_K,
        mmr_lambda=settings.MMR_LAMBDA,
    )

    if not retrieved:
        return Answer(
            text="I don't have that information in the provided documents.",
            used_chunks=[],
            rejected=True,
        )

    best_score = retrieved[0][1]
    if best_score < settings.MIN_SCORE:
        return Answer(
            text="I don't have that information in the provided documents.",
            used_chunks=retrieved,
            rejected=True,
        )

    context = _format_context(retrieved, settings.MAX_CONTEXT_CHARS)

    client = Groq(api_key=settings.GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
        ],
        temperature=0.2,
    )

    text = (resp.choices[0].message.content or "").strip()
    if not text:
        text = "I don't have that information in the provided documents."

    return Answer(text=text, used_chunks=retrieved, rejected=False)
