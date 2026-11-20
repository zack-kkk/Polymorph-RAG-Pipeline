from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, 
    ForeignKey, Boolean, Float, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from src.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Document(Base):
    """Stores source metadata and data lineage."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_hash = Column(String, unique=True, nullable=False, index=True)  # Deduplication
    api_name = Column(String, nullable=False, index=True)  # e.g., Paracetamol, Ibuprofen
    doc_type = Column(String, nullable=False)  # 'patent', 'paper', 'pxrd_csv'
    publication_year = Column(Integer, nullable=True)
    source_owner = Column(String, default="Public Literature / USPTO")
    ingested_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="PROCESSED")  # PROCESSED or QUARANTINED

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    """Stores text chunks, page/row lineage, and pgvector embeddings."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_or_row = Column(String, nullable=False)  # Lineage: exact page or CSV row
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # Matches all-MiniLM-L6-v2

    document = relationship("Document", back_populates="chunks")


class DataQualityLog(Base):
    """Tracks data-quality anomalies (duplicates, unreadable scans, missing metadata)."""
    __tablename__ = "dq_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    issue_type = Column(String, nullable=False)  # DUPLICATE, UNREADABLE_PAGE, MISSING_METADATA
    details = Column(Text, nullable=False)
    logged_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
