from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from `.env`.

    Keep all environment-dependent values in one object so the rest of the
    code does not need to read environment variables directly.
    """

    llm_provider: str
    raw_inputs_dir: Path
    notes_dir: Path
    chroma_dir: Path
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str
    openai_model: str
    openai_embedding_model: str


def load_settings() -> Settings:
    """Load `.env` and return typed settings for the application."""

    load_dotenv()

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "ollama").lower(),
        raw_inputs_dir=Path(os.getenv("RAW_INPUTS_DIR", "raw_inputs")),
        notes_dir=Path(os.getenv("NOTES_DIR", "notes")),
        chroma_dir=Path(os.getenv("CHROMA_DIR", ".chroma")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:14b"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:4b"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
    )
