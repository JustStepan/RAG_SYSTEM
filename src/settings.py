from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    DB_STORAGE: str = f'{BASE_DIR}/storage/'
    CHAT_MEMORY_DIR: str = f'{BASE_DIR}/memory/'
    PDF_DIR: Path = BASE_DIR / "PDF"
    COLLECTION_NAME: str
    OPENAI_API_KEY: str
    LLM_MODEL: str = "qwen3.5-9b"
    EMBEDDING_MODEL: str = "text-embedding-berta-uncased"
    LLM_URL: str = "http://localhost:1234/v1"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 100

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()