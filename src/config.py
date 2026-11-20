import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://admin:password123@localhost:5432/polymorph_db"
    )
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Outputs 384-dimensional vectors
    LLM_MODEL: str = "gpt-4o-mini"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

settings = Settings()
