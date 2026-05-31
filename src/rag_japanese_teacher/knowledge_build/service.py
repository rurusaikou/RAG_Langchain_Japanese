from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_japanese_teacher.core.config import Settings
from rag_japanese_teacher.core.models import build_chat_model


SUPPORTED_INPUT_SUFFIXES = {".md", ".markdown", ".txt"}

# Knowledge build has three raw input sources for now. The keys are directory
# names under `raw_inputs/`; the values are labels shown in CLI output.
INPUT_GROUPS = {
    "class_notes": "上课日语笔记",
    "project_experiences": "项目经历",
    "interview_summaries": "面试问题总结",
}
VALID_CATEGORIES = {
    "vocabulary",
    "grammar",
    "conversation",
    "interview",
    "interview_problem",
    "themes",
    "answer_templates",
}

# Keep the extraction taxonomy close to the product positioning. The knowledge
# base is not a generic Japanese notebook; it is a Japanese IT career-change
# interview coach that combines interview facts with Japanese expression notes.
CATEGORY_GUIDE = """
- vocabulary: 单词、熟语、商务表达、面试关键词。
- grammar: 语法点、句型、容易出错的表达。
- conversation: 自然口语、职场会话、老师纠正过的自然说法。
- interview: 转职理由、志望动机、自我介绍、项目经验、技术问题、强弱项等面试素材。
- interview_problem: 发生过的问题、课题、失败、原因、改善、结果、学到的东西。
- themes: 话题型知识，例如教育、AI、働き方、チーム開発等。
- answer_templates: 可复用回答框架，例如 STAR、転職理由、失敗経験、志望動機模板。
""".strip()

# Long documents should not be sent to a local model in one piece. Chunking
# reduces omission: the model sees a smaller section and can extract more
# complete notes from it.
MAX_CHUNK_CHARS = 2000
CHUNK_OVERLAP_CHARS = 200


@dataclass(frozen=True)
class RawInputFile:
    """A raw source file before it is converted into RAG notes."""

    group: str
    path: Path
    relative_path: Path
    text: str


@dataclass(frozen=True)
class RawInputChunk:
    """A chunk of a raw source file.

    `index` and `total` are included in prompts so generated notes can record
    exactly which part of the source produced them.
    """

    source_file: RawInputFile
    index: int
    total: int
    text: str


@dataclass(frozen=True)
class DraftResult:
    """Result of attempting to write one generated note file."""

    path: Path
    skipped: bool


def scan_raw_inputs(settings: Settings) -> list[RawInputFile]:
    """Find raw class-note and interview-summary files."""

    if not settings.raw_inputs_dir.exists():
        return []

    files: list[RawInputFile] = []
    for group in INPUT_GROUPS:
        group_dir = settings.raw_inputs_dir / group
        if not group_dir.exists():
            continue

        for path in sorted(group_dir.rglob("*")):
            if not path.is_file():
                continue

            # README files describe the directory. They are not source data.
            if path.name.lower() == "readme.md":
                continue
            if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
                continue

            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            files.append(
                RawInputFile(
                    group=group,
                    path=path,
                    relative_path=path.relative_to(settings.raw_inputs_dir),
                    text=text,
                )
            )

    return files


def summarize_raw_inputs(files: list[RawInputFile]) -> dict[str, int]:
    """Count how many raw files exist for each input group."""

    summary = {group: 0 for group in INPUT_GROUPS}
    for file in files:
        summary[file.group] += 1
    return summary


def build_knowledge_drafts(
    settings: Settings,
    source: str = "all",
    dry_run: bool = False,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[DraftResult]:
    """Convert raw input files into draft Markdown notes.

    This is the main knowledge-build pipeline. It scans raw files, chunks long content,
    asks the LLM to extract structured notes, and writes those notes under
    `notes/`.
    """

    files = scan_raw_inputs(settings)
    if source != "all":
        files = [file for file in files if file.group == source]

    if not files:
        return []

    results: list[DraftResult] = []
    for file_index, raw_file in enumerate(files, start=1):
        chunks = split_raw_input_file(raw_file)
        _emit_progress(
            progress,
            f"[{file_index}/{len(files)}] Processing {raw_file.relative_path} "
            f"({len(chunks)} chunk(s))",
        )

        for chunk in chunks:
            # Each chunk becomes a separate LLM call. This is slower than one
            # big call, but it avoids the "long document only summarized a few
            # highlights" problem we observed in early knowledge-building tests.
            _emit_progress(
                progress,
                f"[{file_index}/{len(files)}] Processing {raw_file.relative_path} "
                f"chunk {chunk.index}/{chunk.total}",
            )
            notes = _extract_notes(settings, chunk)
            _emit_progress(
                progress,
                f"[{file_index}/{len(files)}] Chunk {chunk.index}/{chunk.total} "
                f"generated {len(notes)} note draft(s)",
            )

            for note in notes:
                category = note["category"]
                filename = _safe_filename(note["filename"])
                content = note["content"].strip() + "\n"
                output_path = settings.notes_dir / category / filename

                if dry_run:
                    results.append(DraftResult(path=output_path, skipped=False))
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Default behavior is conservative: never overwrite a human-
                # edited note unless the caller explicitly asks for it.
                if output_path.exists() and not overwrite:
                    results.append(DraftResult(path=output_path, skipped=True))
                    _emit_progress(progress, f"Skipped existing {output_path}")
                    continue

                output_path.write_text(content, encoding="utf-8")
                results.append(DraftResult(path=output_path, skipped=False))
                _emit_progress(progress, f"Wrote {output_path}")

    return results

def split_raw_input_file(raw_file: RawInputFile) -> list[RawInputChunk]:
    """Split a long raw file into chunks small enough for local LLM extraction."""

    sections = _split_by_headings(raw_file.text)
    chunks: list[str] = []
    current = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(section))
            continue

        if current and len(current) + len(section) + 2 > MAX_CHUNK_CHARS:
            chunks.append(current.strip())
            overlap = current[-CHUNK_OVERLAP_CHARS:] if len(current) > CHUNK_OVERLAP_CHARS else ""
            current = f"{overlap}\n\n{section}" if overlap else section
        else:
            current = f"{current}\n\n{section}".strip() if current else section

    if current:
        chunks.append(current.strip())

    if not chunks:
        chunks = [raw_file.text]

    total = len(chunks)
    return [
        RawInputChunk(source_file=raw_file, index=index, total=total, text=text)
        for index, text in enumerate(chunks, start=1)
    ]


def _split_by_headings(text: str) -> list[str]:
    """Split text into rough sections using Markdown-like and lesson-style headings."""

    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    heading_pattern = re.compile(r"^\s*(#{1,3}\s+|[【\[]?.{0,30}[：:]$)")
    emoji_heading_pattern = re.compile(r"^\s*[^\w\s]{1,4}\s*\S+")

    for line in lines:
        is_heading = bool(heading_pattern.match(line)) or bool(emoji_heading_pattern.match(line))
        if is_heading and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current).strip())

    return sections


def _split_long_text(text: str) -> list[str]:
    """Fallback splitter for one section that is still too long."""

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks


def _extract_notes(settings: Settings, chunk: RawInputChunk) -> list[dict[str, str]]:
    """Ask the LLM to convert one chunk into structured note drafts."""

    raw_file = chunk.source_file
    extraction_focus = _get_extraction_focus(raw_file.group)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严谨的“日语转职面试知识库”整理助手。"
                "你的任务不是普通总结，而是把原始资料拆成适合 RAG 检索、"
                "并能直接帮助日本 IT 转职面试练习的 Markdown 知识库。"
                "必须只输出 JSON，不要输出解释文字。",
            ),
            (
                "human",
                """
输入来源：{group_label}
原始文件：{source_path}
当前分块：{chunk_index}/{chunk_total}
本来源的抽取重点：
{extraction_focus}

请把下面资料拆分成多个独立知识点，每个知识点生成一个 Markdown 文件。

分类只能使用：
{category_guide}

输出 JSON 格式：
{{
  "notes": [
    {{
      "category": "grammar",
      "filename": "ため.md",
      "content": "# ため\\n\\n## 来源\\n\\n- 上课日语笔记: xxx.md\\n..."
    }}
  ]
}}

整理规则：
- 一个知识点、一个会话场景、一个面试问题、一个问题改善案例，生成一个文件。
- 尽量完整抽取当前分块中的有效内容，不要只总结最重要的 2-3 个点。
- 如果当前分块包含多个问题、多个表达或多个语法点，请分别生成多个文件。
- 如果资料中出现「问题、課題、不具合、ミス、原因、改善、対応、効率化、自動化、失敗、反省」等内容，必须优先整理成单独的 interview_problem 文档。
- 问题解决型 interview_problem 文档必须包含这些小节：## 想定質問、## 事象、## 原因、## 改善・対応、## 結果、## 学び、## 面试回答例、## 追加質問、## 关键词。
- 项目经历类 interview 文档必须尽量包含：## 项目背景、## 担当职责、## 技术栈、## 开发内容、## 成果、## 面试可讲重点、## 日语回答例、## 可能追问、## 来源。
- 面试素材类 interview 文档建议包含：## 想定質問、## 中文原意、## 日语自然表达、## 更商务的表达、## 简短版、## 标准版、## 深挖版、## 相关表达、## 可能追问、## 来源。
- 日语课笔记中的表达如果能用于面试，必须在 content 中加入「## 面试可用表达」或「## 面试使用场景」。
- 如果资料中出现可复用回答结构，整理到 answer_templates，并包含：## 使用场景、## 回答结构、## 日语模板、## 中文说明、## 替换变量、## 注意点。
- 一个问题或一个改善案例只能生成一个独立文件，不要混进转职理由、强み、自我介绍等通用文档。
- content 必须是完整 Markdown。
- content 必须包含「## 来源」。
- 来源中必须写明原始文件和分块编号。
- 上课笔记优先整理到 vocabulary / grammar / conversation / themes；如果能服务面试，也要写清楚面试使用方式。
- 项目经历优先整理到 interview / interview_problem；项目中的技术词、业务词也可以整理到 vocabulary。
- 面试总结优先整理到 interview / interview_problem / answer_templates；也可以提取常用表达到 vocabulary / conversation。
- 不要编造原始资料中没有的信息；可以用「未记录」标注缺失内容。
- 文件名使用日语或中文关键词，必须以 .md 结尾。

原始资料：
```text
{raw_text}
```
""".strip(),
            ),
        ]
    )

    chain = prompt | build_chat_model(settings) | StrOutputParser()
    response = chain.invoke(
        {
            "group_label": INPUT_GROUPS[raw_file.group],
            "source_path": str(raw_file.relative_path),
            "chunk_index": chunk.index,
            "chunk_total": chunk.total,
            "extraction_focus": extraction_focus,
            "category_guide": CATEGORY_GUIDE,
            "raw_text": chunk.text,
        }
    )

    data = _parse_json_response(response)
    notes = data.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError(f"Invalid notes JSON from {raw_file.relative_path}")

    cleaned: list[dict[str, str]] = []
    for note in notes:
        if not isinstance(note, dict):
            continue

        category = str(note.get("category", "")).strip()
        filename = str(note.get("filename", "")).strip()
        content = str(note.get("content", "")).strip()

        if category not in VALID_CATEGORIES:
            continue
        if not filename or not content:
            continue
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        cleaned.append(
            {
                "category": category,
                "filename": filename,
                "content": content,
            }
        )

    return cleaned


def _get_extraction_focus(group: str) -> str:
    """Describe how each raw input group should be converted into notes."""

    if group == "class_notes":
        return (
            "这是日语老师上课笔记。重点抽取单词、语法、自然表达、老师纠错、"
            "话题展开方式，并补充它们在转职面试中的可用场景。"
        )
    if group == "project_experiences":
        return (
            "这是用户真实项目经历。重点抽取项目背景、业务目标、担当职责、"
            "技术栈、开发内容、成果、困难、问题、原因、改善行动、量化结果、"
            "学到的东西，以及可以直接用于日本 IT 转职面试的回答素材。"
        )
    if group == "interview_summaries":
        return (
            "这是用户转职面试准备资料。重点抽取转职理由、志望动机、项目经历、"
            "技术问题、问题改善案例、失败经验、可复用回答模板。"
        )
    return "按日语转职面试知识库的目标抽取可复习、可检索、可回答的内容。"


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    """Report progress only when the CLI provides a callback."""

    if progress:
        progress(message)


def _parse_json_response(response: str) -> dict:
    """Parse JSON from an LLM response.

    Models sometimes wrap JSON in markdown fences. This accepts both plain JSON
    and fenced JSON.
    """

    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _safe_filename(filename: str) -> str:
    """Normalize model-generated filenames before writing to disk."""

    filename = filename.strip().replace("/", "_").replace("\\", "_")
    filename = re.sub(r"\s+", "_", filename)
    filename = re.sub(r"[\x00-\x1f]", "", filename)
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    return filename
