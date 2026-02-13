from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from rag.types import Document


def load_pdf_pages(pdf_path: Path, *, company_id: str) -> List[Document]:
    reader = PdfReader(str(pdf_path))
    docs: List[Document] = []

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        docs.append(
            Document(
                text=text,
                metadata={
                    "company_id": company_id,
                    "source": str(pdf_path.name),
                    "page": i + 1,
                },
            )
        )
    return docs
