from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database import SessionLocal, DataQualityLog, Document, init_db
from src.rag import retrieve_chunks, generate_grounded_answer

app = FastAPI(
    title="Polymorph Literature RAG API",
    description="Cited Q&A over open-access API polymorph literature and PXRD tables with lineage tracking.",
    version="1.0.0"
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class QueryRequest(BaseModel):
    question: str
    api_name: Optional[str] = None
    top_k: int = 4


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    doc_count = db.query(Document).count()
    return {"status": "healthy", "indexed_documents": doc_count}


@app.post("/query")
def query_polymorph_literature(req: QueryRequest, db: Session = Depends(get_db)):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    chunks = retrieve_chunks(db, query=req.question, api_name=req.api_name, top_k=req.top_k)
    result = generate_grounded_answer(query=req.question, chunks=chunks)
    return result


@app.get("/dq-report")
def get_data_quality_report(db: Session = Depends(get_db)):
    logs = db.query(DataQualityLog).order_by(DataQualityLog.logged_at.desc()).limit(50).all()
    return {
        "total_issues": len(logs),
        "issues": [
            {
                "file_name": log.file_name,
                "issue_type": log.issue_type,
                "details": log.details,
                "logged_at": log.logged_at
            }
            for log in logs
        ]
    }
