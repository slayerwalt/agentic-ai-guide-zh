"""Read the book's active TeX headings without counting commented translations."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER_SLUGS = (
    "llm-architecture", "systems-foundations", "reinforcement-learning",
    "rl-for-language-models", "ppo", "dpo", "grpo", "preference-optimization",
    "reward-models", "sft", "training-infrastructure", "agentic-training",
    "reasoning", "evaluation", "agentic-ai", "rag", "memory", "agent-harness",
    "loop-engineering", "agent-patterns", "environments", "mcp", "agent-skills",
    "a2a", "multi-agent-systems", "development-frameworks", "agentic-ui",
    "quiz", "quick-reference", "conclusion",
)
PART_SLUGS = (
    "foundations", "rl-methods", "reasoning", "evaluation", "agentic-ai",
    "assessment-reference",
)


def active_tex(path: Path) -> str:
    """Keep source line positions; escaped percent signs are ordinary content."""
    lines = []
    verbatim = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if verbatim:
            lines.append(line)
            if rf"\end{{{verbatim}}}" in line:
                verbatim = None
            continue
        for index, char in enumerate(line):
            if char == "%":
                preceding = len(line[:index]) - len(line[:index].rstrip("\\"))
                if preceding % 2 == 0:
                    # A listings environment is verbatim: Python's modulo,
                    # format specifiers and comments must survive unchanged.
                    if re.search(r"\\begin\{(?:lstlisting|verbatim)\}", line[:index]):
                        break
                    line = line[:index]
                    break
        opened = re.search(r"\\begin\{(lstlisting|verbatim)\}", line)
        if opened and rf"\end{{{opened[1]}}}" not in line:
            verbatim = opened[1]
        lines.append(line)
    return "\n".join(lines)


def braced(text: str, start: int) -> tuple[str, int]:
    depth = 0
    for index in range(start, len(text)):
        if text[index] in "{}" and (index == 0 or text[index - 1] != "\\"):
            depth += 1 if text[index] == "{" else -1
            if depth == 0:
                return text[start + 1:index], index + 1
    raise ValueError(f"Unbalanced heading near {text[start:start + 80]!r}")


def plain_title(value: str) -> str:
    # The first argument is the visible heading; the second is a PDF bookmark.
    while match := re.search(r"\\texorpdfstring(?=\{)", value):
        visible, end = braced(value, match.end())
        while value[end].isspace():
            end += 1
        _, finish = braced(value, end)
        value = value[:match.start()] + visible + value[finish:]
    while re.search(r"\\(?:textbf|textit|texttt|emph)\{([^{}]*)\}", value):
        value = re.sub(r"\\(?:textbf|textit|texttt|emph)\{([^{}]*)\}", r"\1", value)
    return (value.replace(r"\&", "&").replace(r"\_", "_")
            .replace(r"\%", "%").replace("~", " ").replace("---", "—")
            .replace("--", "–").strip())


def headings(text: str):
    pattern = r"^\\(part|chapter|section|subsection|subsubsection)(\*?)\s*(?=\{)"
    for match in re.finditer(pattern, text, re.M):
        title, end = braced(text, match.end())
        label = re.match(r"\s*\\label\{([^}]+)\}", text[end:])
        yield {
            "kind": match[1], "starred": bool(match[2]),
            "title": plain_title(title), "label": label[1] if label else None,
            "line": text.count("\n", 0, match.start()) + 1,
        }


def catalog() -> dict:
    front = [h for h in headings(active_tex(ROOT / "book.tex"))
             if h["kind"] == "chapter" and h["starred"]]
    parts = []
    chapter_number = 0
    environments: Counter = Counter()
    environments.update(re.findall(r"\\begin\{([^}]+)\}", active_tex(ROOT / "book.tex")))
    for part_number, relative in enumerate(
        re.findall(r"^\\input\{(contents/[^}]+)\}", active_tex(ROOT / "book.tex"), re.M), 1
    ):
        source = relative + ".tex"
        text = active_tex(ROOT / source)
        environments.update(re.findall(r"\\begin\{([^}]+)\}", text))
        part = None
        chapter = None
        for heading in headings(text):
            if heading["kind"] == "part":
                part = dict(heading, number=part_number, source=source,
                            path=f"parts/{part_number:02d}-{PART_SLUGS[part_number - 1]}/index.md",
                            chapters=[])
                parts.append(part)
            elif heading["kind"] == "chapter" and not heading["starred"]:
                chapter_number += 1
                chapter = dict(
                    heading, number=chapter_number, source=source, sections=[],
                    path=f"chapters/{chapter_number:02d}-{CHAPTER_SLUGS[chapter_number - 1]}/index.md",
                )
                part["chapters"].append(chapter)
            elif heading["kind"] == "section" and not heading["starred"]:
                number = f"{chapter_number}.{len(chapter['sections']) + 1}"
                chapter["sections"].append(dict(heading, number=number))
    if len(parts) != 6 or chapter_number != len(CHAPTER_SLUGS):
        raise ValueError("Book structure changed; review the chapter routes before scaffolding.")
    return {
        "front_matter": front, "parts": parts,
        "section_count": sum(len(c["sections"]) for p in parts for c in p["chapters"]),
        "environments": dict(environments),
        "bibliography_entries": len(re.findall(
            r"^@\w+\s*\{", (ROOT / "book.bib").read_text(encoding="utf-8"), re.M)),
    }


if __name__ == "__main__":
    print(json.dumps(catalog(), ensure_ascii=False, indent=2))
