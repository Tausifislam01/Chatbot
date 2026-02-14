from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.types import Document


def mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    doc_ids: List[int],
    *,
    top_k: int,
    lambda_mult: float = 0.6,
) -> List[int]:

    selected: List[int] = []
    candidates = doc_ids[:]

    q = query_vec.reshape(1, -1)
    sim_to_query = (doc_vecs[candidates] @ q.T).reshape(-1)  

    while candidates and len(selected) < top_k:
        if not selected:
            best_idx = int(np.argmax(sim_to_query))
            selected.append(candidates.pop(best_idx))
            sim_to_query = np.delete(sim_to_query, best_idx)
            continue

        selected_vecs = doc_vecs[selected]
        sim_to_selected = doc_vecs[candidates] @ selected_vecs.T  
        max_sim_to_selected = sim_to_selected.max(axis=1)

        mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected
        best_idx = int(np.argmax(mmr_score))

        selected.append(candidates.pop(best_idx))
        sim_to_query = np.delete(sim_to_query, best_idx)

    return selected


def retrieve(
    *,
    embed_model: SentenceTransformer,
    faiss_index,
    chunks: List[Document],
    query: str,
    top_k: int = 8,
    mmr_lambda: float = 0.6,
) -> List[Tuple[Document, float]]:
    q_vec = embed_model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype=np.float32)

    scores, idxs = faiss_index.search(q_vec, top_k * 4) 
    scores = scores[0].tolist()
    idxs = idxs[0].tolist()

    valid = [(i, s) for i, s in zip(idxs, scores) if i != -1]
    if not valid:
        return []

    cand_ids = [i for i, _ in valid]
    doc_vecs = np.vstack([faiss_index.reconstruct(i) for i in cand_ids]).astype(np.float32)

    selected_local = mmr(
        query_vec=q_vec[0],
        doc_vecs=doc_vecs,
        doc_ids=list(range(len(cand_ids))),
        top_k=min(top_k, len(cand_ids)),
        lambda_mult=mmr_lambda,
    )

    results: List[Tuple[Document, float]] = []
    for local_id in selected_local:
        global_id = cand_ids[local_id]
        score = dict(valid)[global_id]
        results.append((chunks[global_id], float(score)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
