from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Callable

from rag_japanese_teacher.core.config import Settings
from rag_japanese_teacher.core.models import build_chat_model, build_embeddings
from rag_japanese_teacher.knowledge.documents import (
    format_documents,
    format_sources,
    load_markdown_documents,
)
from rag_japanese_teacher.rag.prompts import BASE_SYSTEM_PROMPT, get_mode_prompt


COLLECTION_NAME = "japanese_teacher_notes"


def build_vectorstore(settings: Settings) -> Chroma:
    """Open the local Chroma vector store."""

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
    )


def ingest_notes(settings: Settings, progress: Callable[[str], None] | None = None) -> int:
    """Embed all finalized notes and store them in Chroma."""

    _emit_progress(progress, f"Loading Markdown notes from {settings.notes_dir}")
    documents = load_markdown_documents(settings.notes_dir)
    _emit_progress(progress, f"Loaded {len(documents)} Markdown note(s)")
    _emit_progress(progress, "Creating embedding model")
    embeddings = build_embeddings(settings)
    _emit_progress(progress, f"Embedding notes and writing Chroma index to {settings.chroma_dir}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(settings.chroma_dir),
    )

    persist = getattr(vectorstore, "persist", None)
    if callable(persist):
        _emit_progress(progress, "Persisting Chroma index")
        persist()

    _emit_progress(progress, f"Indexed {len(documents)} Markdown note(s)")
    return len(documents)


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    """Report progress only when the caller provides a callback."""

    if progress:
        progress(message)


def answer_question(settings: Settings, question: str, mode: str = "general") -> tuple[str, str]:
    """Answer a user question with RAG.

    Flow:
    1. Retrieve related notes from Chroma.
    2. Put those notes into the prompt as context.
    3. Ask the configured LLM to answer in the selected learning mode.
    """

    vectorstore = build_vectorstore(settings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    documents = retriever.invoke(question)

    context = format_documents(documents)
    sources = format_sources(documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", BASE_SYSTEM_PROMPT),
            ("system", "{mode_prompt}"),
            (
                "human",
                "下面是用户个人日语笔记中检索到的内容：\n\n{context}\n\n"
                "用户问题：\n{question}",
            ),
        ]
    )

    llm = build_chat_model(settings)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "mode_prompt": get_mode_prompt(mode),
            "context": context,
            "question": question,
        }
    )

    return answer, sources
