from __future__ import annotations

import re
from pathlib import Path
from typing import Any

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def chunk_markdown_text(
    text: str,
    source: str,
) -> list[dict[str, Any]]:
    """Split Markdown text into hierarchical heading-based chunks."""

    if not text.strip():
        raise ValueError("text must not be empty.")

    lines = text.splitlines()

    chunks: list[dict[str, Any]] = []

    heading_stack: list[tuple[int, str]] = []

    current_heading: str | None = None
    current_level: int | None = None
    current_section_path: str | None = None
    current_lines: list[str] = []

    def flush_chunk() -> None:
        if current_heading is None:
            return

        content = "\n".join(current_lines).strip()

        if not content:
            return

        chunks.append(
            {
                "source": source,
                "heading": current_heading,
                "heading_level": current_level,
                "section_path": current_section_path,
                "content": content,
            }
        )

    for line in lines:
        match = HEADING_PATTERN.match(line.strip())

        if not match:
            current_lines.append(line)
            continue

        flush_chunk()

        level = len(match.group(1))
        heading = match.group(2).strip()

        heading_stack = [
            (stack_level, stack_heading)
            for stack_level, stack_heading in heading_stack
            if stack_level < level
        ]

        heading_stack.append(
            (level, heading)
        )

        current_heading = heading
        current_level = level
        current_section_path = " > ".join(
            stack_heading
            for _, stack_heading in heading_stack
        )
        current_lines = []

    flush_chunk()

    return chunks


def chunk_markdown_file(
    path: Path,
) -> list[dict[str, Any]]:
    """Load and chunk a Markdown file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Markdown file not found: {path}"
        )

    if path.suffix.lower() != ".md":
        raise ValueError(
            f"Expected a Markdown file, got: {path}"
        )

    text = path.read_text(
        encoding="utf-8"
    )

    return chunk_markdown_text(
        text=text,
        source=path.name,
    )


def chunk_knowledge_directory(
    knowledge_dir: Path,
) -> list[dict[str, Any]]:
    """Chunk all Markdown files in the knowledge directory."""

    if not knowledge_dir.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {knowledge_dir}"
        )

    markdown_files = sorted(
        knowledge_dir.glob("*.md")
    )

    if not markdown_files:
        raise ValueError(
            f"No Markdown files found in: {knowledge_dir}"
        )

    chunks: list[dict[str, Any]] = []

    for path in markdown_files:
        chunks.extend(
            chunk_markdown_file(path)
        )

    for chunk_id, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk_id

    return chunks