import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_japanese_teacher.core.config import load_settings
from rag_japanese_teacher.knowledge_build.service import (
    INPUT_GROUPS,
    build_knowledge_drafts,
    scan_raw_inputs,
    split_raw_input_file,
    summarize_raw_inputs,
)
from rag_japanese_teacher.rag.service import answer_question, ingest_notes


console = Console()


def run_ingest() -> int:
    """CLI handler for `jp-teacher ingest`."""

    settings = load_settings()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Preparing ingest...", total=None)

        def report(message: str) -> None:
            """Bridge RAG ingest progress messages into Rich progress UI."""

            progress.update(task, description=message)

        count = ingest_notes(settings, progress=report)

    console.print(f"[green]Indexed {count} Markdown notes.[/green]")
    console.print(f"Vector database: [bold]{settings.chroma_dir}[/bold]")
    return 0


def run_ask(question: str | None, mode: str) -> int:
    """CLI handler for one-shot and interactive question answering."""

    settings = load_settings()

    if question:
        answer, sources = answer_question(settings, question, mode)
        console.print(Panel(Markdown(answer), title="AI 日语转职面试老师"))
        if sources:
            console.print(Panel(sources, title="参考笔记"))
        return 0

    console.print("[bold]进入交互模式。输入 exit / quit 结束。[/bold]")
    while True:
        try:
            user_input = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0

        if user_input.lower() in {"exit", "quit"}:
            return 0
        if not user_input:
            continue

        answer, sources = answer_question(settings, user_input, mode)
        console.print(Panel(Markdown(answer), title="AI 日语转职面试老师"))
        if sources:
            console.print(Panel(sources, title="参考笔记"))


def run_knowledge_build_scan() -> int:
    """CLI handler for scanning raw knowledge-build inputs."""

    settings = load_settings()
    files = scan_raw_inputs(settings)
    summary = summarize_raw_inputs(files)

    table = Table(title="Knowledge Build 原始输入扫描")
    table.add_column("输入类型")
    table.add_column("目录")
    table.add_column("文件数", justify="right")

    for group, label in INPUT_GROUPS.items():
        table.add_row(label, str(settings.raw_inputs_dir / group), str(summary[group]))

    console.print(table)

    if not files:
        console.print(
            "[yellow]没有发现原始输入文件。请把 .md / .txt 文件放入 "
            "raw_inputs/class_notes/、raw_inputs/project_experiences/ "
            "或 raw_inputs/interview_summaries/。[/yellow]"
        )
        return 0

    console.print("\n[bold]发现文件：[/bold]")
    for file in files:
        chunks = split_raw_input_file(file)
        console.print(f"- {file.relative_path} ({len(chunks)} chunk(s))")
    return 0


def run_knowledge_build(source: str, dry_run: bool, overwrite: bool) -> int:
    """CLI handler for converting raw inputs into structured knowledge notes."""

    settings = load_settings()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Preparing knowledge build...", total=None)

        def report(message: str) -> None:
            """Bridge service-layer progress messages into Rich progress UI."""

            progress.update(task, description=message)

        results = build_knowledge_drafts(
            settings=settings,
            source=source,
            dry_run=dry_run,
            overwrite=overwrite,
            progress=report,
        )

    if not results:
        console.print(
            "[yellow]没有可处理的原始输入文件。先运行 `jp-teacher knowledge_build scan` 查看。[/yellow]"
        )
        return 0

    for result in results:
        if result.skipped:
            console.print(f"[yellow]Skipped existing:[/yellow] {result.path}")
        elif dry_run:
            console.print(f"[cyan]Would write:[/cyan] {result.path}")
        else:
            console.print(f"[green]Wrote:[/green] {result.path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Define all supported `jp-teacher` commands and options."""

    parser = argparse.ArgumentParser(
        description="Personal Japanese teacher powered by LangChain + RAG."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build or refresh the local vector index.")

    knowledge_build_parser = subparsers.add_parser(
        "knowledge_build",
        aliases=["knowledge-build"],
        help="Build structured notes from class notes, project experiences, and interview summaries.",
    )
    knowledge_build_subparsers = knowledge_build_parser.add_subparsers(
        dest="knowledge_build_command",
        required=True,
    )

    knowledge_build_subparsers.add_parser("scan", help="Scan raw input files.")

    knowledge_build_run_parser = knowledge_build_subparsers.add_parser(
        "build",
        help="Convert raw input files into notes/ Markdown drafts.",
    )
    knowledge_build_run_parser.add_argument(
        "--source",
        choices=["all", "class_notes", "project_experiences", "interview_summaries"],
        default="all",
        help="Which raw input source to process.",
    )
    knowledge_build_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be written without writing them.",
    )
    knowledge_build_run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing notes if generated file names already exist.",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask your AI Japanese teacher.")
    ask_parser.add_argument("question", nargs="?", help="Question or practice prompt.")
    ask_parser.add_argument(
        "--mode",
        choices=["vocabulary", "grammar", "conversation", "interview", "theme", "general"],
        default="general",
        help="Learning mode.",
    )

    return parser


def main() -> int:
    """Program entry point configured in `pyproject.toml`."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "ingest":
            return run_ingest()
        if args.command in {"knowledge_build", "knowledge-build"}:
            if args.knowledge_build_command == "scan":
                return run_knowledge_build_scan()
            if args.knowledge_build_command == "build":
                return run_knowledge_build(args.source, args.dry_run, args.overwrite)
        if args.command == "ask":
            return run_ask(args.question, args.mode)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
