from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.types import Document


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    # vectors already normalized -> inner product == cosine similarity
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_store(out_dir: Path, index: faiss.Index, chunks: List[Document]) -> None:
    _ensure_dir(out_dir)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")


def load_store(store_dir: Path) -> Tuple[faiss.Index, List[Document]]:
    index = faiss.read_index(str(store_dir / "index.faiss"))
    chunks: List[Document] = []
    with (store_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(Document(text=obj["text"], metadata=obj["metadata"]))
    return index, chunks
