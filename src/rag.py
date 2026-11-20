from typing import Optional
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from src.database import Document, DocumentChunk
from src.config import settings

embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
llm_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None


def retrieve_chunks(
    db: Session,
    query: str,
    api_name: Optional[str] = None,
    top_k: int = 4
) -> list[dict]:
    query_vector = embedder.encode(query).tolist()

    stmt = (
        db.query(
            DocumentChunk,
            Document,
            DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .filter(Document.status == "PROCESSED")
    )

    if api_name:
        stmt = stmt.filter(Document.api_name.ilike(f"%{api_name}%"))

    results = stmt.order_by("distance").limit(top_k).all()

    retrieved = []
    for chunk, doc, distance in results:
        retrieved.append({
            "chunk_id": chunk.id,
            "file_name": doc.file_name,
            "api_name": doc.api_name,
            "year": doc.publication_year,
            "location": chunk.page_or_row,
            "similarity": round(1 - float(distance), 4),
            "content": chunk.content
        })
    return retrieved


def generate_grounded_answer(query: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "Insufficient evidence in the indexed polymorph literature to answer this question.",
            "citations": []
        }

    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i}: {c['file_name']} | {c['location']} | Year: {c['year']}]\n{c['content']}"
        )
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a pharmaceutical solid-state chemistry assistant. "
        "Answer the user's question strictly using ONLY the provided context. "
        "Every claim (such as polymorphic forms, PXRD 2-theta peaks, DSC melting points, or crystallization solvents) "
        "MUST include inline citations like [Source 1]. "
        "If the context does not contain the answer, state that clearly. Do not extrapolate."
    )

    if not llm_client:
        # Fallback mode when running offline or without an API key
        return {
            "answer": f"[OFFLINE RETRIEVAL MODE] Top matching excerpt: {chunks[0]['content']} [Source 1]",
            "citations": chunks
        }

    response = llm_client.chat.completions.create(
        model=settings.LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "citations": chunks
    }
