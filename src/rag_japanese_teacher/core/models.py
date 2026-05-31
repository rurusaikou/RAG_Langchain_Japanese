from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_japanese_teacher.core.config import Settings


def build_embeddings(settings: Settings) -> Embeddings:
    """Create the embedding model used by Chroma.

    Phase 1 uses embeddings to convert notes and user questions into vectors.
    Keeping provider selection here lets us switch between Ollama and OpenAI
    without changing the RAG code.
    """

    if settings.llm_provider == "ollama":
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
    if settings.llm_provider == "openai":
        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model used for knowledge extraction and RAG answers."""

    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )
    if settings.llm_provider == "openai":
        return ChatOpenAI(model=settings.openai_model, temperature=0.3)

    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
