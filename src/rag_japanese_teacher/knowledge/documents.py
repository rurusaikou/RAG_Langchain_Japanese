from pathlib import Path

from langchain_core.documents import Document


def load_markdown_documents(notes_dir: Path) -> list[Document]:
    """Load finalized RAG notes from `notes/`.

    The project uses one Markdown file as one knowledge chunk. This is better
    for the current learning notes than blind character splitting because each
    file already represents one word, grammar point, scene, or interview topic.
    """

    if not notes_dir.exists():
        raise FileNotFoundError(f"Notes directory not found: {notes_dir}")

    documents: list[Document] = []
    for path in sorted(notes_dir.rglob("*.md")):
        relative_path = path.relative_to(notes_dir)

        # Directories such as `notes/_templates/` are for humans and should not
        # become searchable knowledge.
        if any(part.startswith("_") for part in relative_path.parts):
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(relative_path),
                    "category": category,
                    "title": path.stem,
                },
            )
        )

    if not documents:
        raise ValueError(f"No Markdown notes found in: {notes_dir}")

    return documents


def format_documents(documents: list[Document]) -> str:
    """Format retrieved notes as context for the LLM."""

    chunks = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        chunks.append(f"[参考笔记 {index}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


def format_sources(documents: list[Document]) -> str:
    """Format unique source paths so the CLI can show what was referenced."""

    lines = []
    seen = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source in seen:
            continue
        seen.add(source)
        title = doc.metadata.get("title", source)
        lines.append(f"- {title}: notes/{source}")
    return "\n".join(lines)
