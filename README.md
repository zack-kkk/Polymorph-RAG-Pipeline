# Polymorph Literature RAG & Data-Quality Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) and ETL data-engineering pipeline designed for pharmaceutical solid-state chemistry. It ingests open-access polymorph research papers, patents (PDF), and PXRD peak tables (CSV), enforces automated data-quality checks, tracks document lineage in PostgreSQL (`pgvector`), and serves cited answers via FastAPI.

## Architecture
1. **ETL & Data Quality (`src/etl.py`)**: Extracts text from PDFs and CSVs, enforces SHA-256 deduplication, flags scanned/unreadable pages (`<50` chars), validates `<APIName>_<DocType>_<Year>_<Title>` metadata schemas, and logs anomalies to `dq_audit_logs`.
2. **Metadata, Lineage & Vector Store (`src/database.py`)**: Stores document-level provenance (API name, doc type, publication year, ingestion timestamp) linked to chunk-level page/row citations and 384-d `pgvector` embeddings.
3. **Grounded RAG API (`app.py`, `src/rag.py`)**: Performs metadata-filtered cosine similarity search and generates strictly cited answers.
4. **CI/CD Regression Gate (`eval/`, `.github/workflows/rag_eval.yml`)**: Runs a Pytest evaluation benchmark against a live `pgvector` service container on every push to enforce retrieval hit-rate and groundedness thresholds.

## Quickstart

### 1. Start PostgreSQL with pgvector
```bash
docker-compose up -d
