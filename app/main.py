from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import answer_question
from app.settings import Settings


settings = Settings()
app = FastAPI(title="Mysoft Heaven RAG Chatbot")

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/chat")
def chat(req: ChatRequest):
    ans = answer_question(settings, req.question)
    return {
        "answer": ans.text,
        "rejected": ans.rejected,
        "sources": [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "score": score,
            }
            for d, score in ans.used_chunks
        ],
    }
