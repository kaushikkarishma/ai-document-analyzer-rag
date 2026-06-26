from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central place for paths and model names used by the backend.

    Keeping these values here avoids hiding important project choices inside
    random functions. In an interview, this is where you can point when asked
    which LLM, embedding model, and local storage paths the app uses.
    """

    app_name: str = "AI Document Analyzer"
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    project_root: Path = Path(__file__).resolve().parents[2]
    upload_dir: Path = project_root / "data" / "uploads"
    chroma_dir: Path = project_root / "data" / "chroma"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOC_ANALYZER_")


settings = Settings()
