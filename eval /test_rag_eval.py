import json
import pytest
from src.database import SessionLocal, Document, DocumentChunk, init_db
from src.etl import embedder
from src.rag import retrieve_chunks, generate_grounded_answer


@pytest.fixture(scope="module")
def seeded_db():
    init_db()
    db = SessionLocal()
    with open("eval/golden_dataset.json", "r") as f:
        data = json.load(f)

    # Seed golden documents if not already present
    for idx, doc_data in enumerate(data["seed_documents"]):
        fake_hash = f"eval_seed_hash_{idx}"
        if not db.query(Document).filter(Document.file_hash == fake_hash).first():
            doc = Document(
                file_name=doc_data["file_name"],
                file_hash=fake_hash,
                api_name=doc_data["api_name"],
                doc_type=doc_data["doc_type"],
                publication_year=doc_data["publication_year"],
                status="PROCESSED"
            )
            db.add(doc)
            db.flush()

            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=0,
                page_or_row=doc_data["page_or_row"],
                content=doc_data["content"],
                embedding=embedder.encode(doc_data["content"]).tolist()
            )
            db.add(chunk)
    db.commit()
    yield db, data["test_cases"]
    db.close()


def test_retrieval_hit_rate_and_groundedness(seeded_db):
    db, test_cases = seeded_db
    hits = 0
    grounded_passes = 0

    for case in test_cases:
        chunks = retrieve_chunks(
            db,
            query=case["question"],
            api_name=case.get("api_name"),
            top_k=3
        )
        retrieved_files = [c["file_name"] for c in chunks]
        combined_text = " ".join([c["content"].lower() for c in chunks])

        # 1. Check Source Retrieval Hit
        if case["expected_source"] in retrieved_files:
            hits += 1

        # 2. Check Groundedness (all required chemical facts present in context)
        if all(kw.lower() in combined_text for kw in case["expected_keywords"]):
            grounded_passes += 1

        # 3. Verify Answer Generation includes citations
        response = generate_grounded_answer(case["question"], chunks)
        assert "[Source 1]" in response["answer"]

    hit_rate = (hits / len(test_cases)) * 100
    grounded_rate = (grounded_passes / len(test_cases)) * 100

    print(f"\n[EVAL METRICS] Retrieval Hit Rate: {hit_rate:.1f}% | Groundedness: {grounded_rate:.1f}%")

    # Regression Thresholds (Fail CI build if quality drops)
    assert hit_rate >= 85.0, f"Retrieval hit rate {hit_rate}% fell below 85% threshold!"
    assert grounded_rate >= 85.0, f"Groundedness rate {grounded_rate}% fell below 85% threshold!"
