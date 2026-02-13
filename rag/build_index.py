from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag.chunking import chunk_documents
from rag.embed_store import build_faiss_index, embed_texts, save_store
from rag.ingest import load_pdf_pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--company-id", required=True, type=str)
    ap.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5", type=str)
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)

    pages = load_pdf_pages(pdf_path, company_id=args.company_id)
    chunks = chunk_documents(pages, target_chars=1200, overlap_chars=200)

    model = SentenceTransformer(args.embed_model)
    vectors = embed_texts(model, [c.text for c in chunks])
    index = build_faiss_index(vectors)

    save_store(out_dir, index, chunks)
    print(f"Saved FAISS store to: {out_dir}")


if __name__ == "__main__":
    main()
