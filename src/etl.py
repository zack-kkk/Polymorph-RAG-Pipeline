import os
import re
import hashlib
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from src.database import SessionLocal, Document, DocumentChunk, DataQualityLog, init_db
from src.config import settings

embedder = SentenceTransformer(settings.EMBEDDING_MODEL)


def compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    # Remove excessive whitespace and hyphenation breaks from PDFs
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def log_dq_issue(db: Session, file_name: str, issue_type: str, details: str):
    log = DataQualityLog(file_name=file_name, issue_type=issue_type, details=details)
    db.add(log)
    db.commit()
    print(f"[DQ FLAG] {file_name} | {issue_type}: {details}")


def parse_filename_metadata(file_name: str) -> dict:
    """
    Expected naming convention: <APIName>_<DocType>_<Year>_<Title>.pdf/csv
    Example: Paracetamol_Patent_2019_US10234567.pdf
    """
    parts = file_name.rsplit(".", 1)[0].split("_")
    if len(parts) < 3 or not parts[2].isdigit():
        return {"valid": False}
    return {
        "valid": True,
        "api_name": parts[0].capitalize(),
        "doc_type": parts[1].lower(),
        "year": int(parts[2])
    }


def ingest_file(file_path: str, db: Session):
    file_name = os.path.basename(file_path)
    file_hash = compute_file_hash(file_path)

    # 1. DQ Check: Duplicate File
    if db.query(Document).filter(Document.file_hash == file_hash).first():
        log_dq_issue(db, file_name, "DUPLICATE_FILE", "File SHA-256 hash already exists in DB.")
        return

    # 2. DQ Check: Metadata completeness
    meta = parse_filename_metadata(file_name)
    if not meta["valid"]:
        log_dq_issue(
            db, file_name, "MISSING_METADATA",
            "Filename must follow <APIName>_<DocType>_<Year>_<Title> format."
        )
        return

    extracted_segments = []  # List of tuples: (page_or_row_label, text)

    # 3. Extraction & Page-Level DQ Check
    if file_name.lower().endswith(".pdf"):
        doc_pdf = fitz.open(file_path)
        for page_num in range(len(doc_pdf)):
            raw_text = doc_pdf[page_num].get_text("text")
            cleaned = clean_text(raw_text)
            # DQ Check: Unreadable / Scanned image page without OCR text
            if len(cleaned) < 50:
                log_dq_issue(
                    db, file_name, "UNREADABLE_PAGE",
                    f"Page {page_num + 1} has < 50 characters (likely scanned image)."
                )
                continue
            extracted_segments.append((f"Page {page_num + 1}", cleaned))

    elif file_name.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        if df.empty:
            log_dq_issue(db, file_name, "EMPTY_CSV", "CSV contains 0 rows.")
            return
        for idx, row in df.iterrows():
            # Convert CSV row (e.g., PXRD peak table) into natural text
            row_text = "; ".join([f"{col}: {val}" for col, val in row.dropna().items()])
            extracted_segments.append((f"Row {idx + 1}", row_text))

    if not extracted_segments:
        log_dq_issue(db, file_name, "NO_VALID_CONTENT", "No extractable text passed DQ checks.")
        return

    # 4. Store Document Lineage
    db_doc = Document(
        file_name=file_name,
        file_hash=file_hash,
        api_name=meta["api_name"],
        doc_type=meta["doc_type"],
        publication_year=meta["year"],
        status="PROCESSED"
    )
    db.add(db_doc)
    db.flush()  # Get db_doc.id

    # 5. Chunk & Embed
    chunk_records = []
    chunk_counter = 0
    for location_label, segment_text in extracted_segments:
        for ch in chunk_text(segment_text):
            vector = embedder.encode(ch).tolist()
            chunk_records.append(
                DocumentChunk(
                    document_id=db_doc.id,
                    chunk_index=chunk_counter,
                    page_or_row=location_label,
                    content=ch,
                    embedding=vector
                )
            )
            chunk_counter += 1

    db.bulk_save_objects(chunk_records)
    db.commit()
    print(f"[INGESTED] {file_name} -> {len(chunk_records)} chunks stored.")


def run_pipeline(data_dir: str = "data/raw"):
    init_db()
    db = SessionLocal()
    try:
        for fname in os.listdir(data_dir):
            if fname.lower().endswith((".pdf", ".csv")):
                ingest_file(os.path.join(data_dir, fname), db)
    finally:
        db.close()


if __name__ == "__main__":
    run_pipeline()
