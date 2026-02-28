"""Generate topic-oriented QA subsets from add_qa_data.py."""
from __future__ import annotations

from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
ADD_QA_PATH = ROOT / "add_qa_data.py"
DATASET_DIR = ROOT / "dataset_topics"


def _is_topic_title(line: str) -> bool:
    candidate = line.strip()
    if not candidate or ":" in candidate:
        return False
    letters = [ch for ch in candidate if ch.isalpha()]
    if not letters:
        return False
    return all(ch.isupper() for ch in letters)


def _parse_topics() -> list[tuple[str, int]]:
    with ADD_QA_PATH.open("r", encoding="utf-8") as fh:
        lines = fh.readlines()

    topics: list[tuple[str, int]] = []
    current_topic: str | None = None
    current_count = 0
    expect_topic = False
    inside = False

    for line in lines:
        stripped = line.strip()
        if not inside:
            if stripped.startswith("QA_DATA") and "[" in stripped:
                inside = True
            continue

        if stripped.startswith("# ============================================================"):
            expect_topic = True
            continue

        if expect_topic and stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if _is_topic_title(candidate):
                if current_topic:
                    topics.append((current_topic, current_count))
                current_topic = candidate
                current_count = 0
                expect_topic = False
                continue
            expect_topic = False

        if stripped.startswith("{"):
            current_count += 1

    if current_topic:
        topics.append((current_topic, current_count))

    return topics


def _load_data() -> list[dict[str, str]]:
    text = ADD_QA_PATH.read_text(encoding="utf-8")
    marker = "QA_DATA"
    start_idx = text.find(marker)
    if start_idx == -1:
        raise RuntimeError("QA_DATA block not found")

    list_start = text.find("[", start_idx)
    if list_start == -1:
        raise RuntimeError("QA_DATA list start not detected")

    depth = 0
    end_idx = None
    for idx in range(list_start, len(text)):
        char = text[idx]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                end_idx = idx + 1
                break

    if end_idx is None:
        raise RuntimeError("QA_DATA list end not detected")

    snippet = f"QA_DATA = {text[list_start:end_idx]}"
    namespace: dict[str, list[dict[str, str]]] = {}
    exec(snippet, namespace)
    return namespace["QA_DATA"]


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_") or "topic"


def main() -> None:
    topics = _parse_topics()
    qa_data = _load_data()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    cursor = 0
    overview: list[dict[str, str]] = []
    exported_total = 0
    for topic_name, count in topics:
        block = qa_data[cursor : cursor + count]
        cursor += count
        slug = _slugify(topic_name)
        filepath = DATASET_DIR / f"{slug}.json"
        with filepath.open("w", encoding="utf-8") as fh:
            json.dump(block, fh, ensure_ascii=False, indent=2)
        overview.append({
            "topic": topic_name,
            "file": filepath.name,
            "samples": len(block),
        })
        exported_total += len(block)

    if exported_total != len(qa_data):
        print(
            "WARNING: actual exported samples differ from QA_DATA length",
            file=sys.stderr,
        )
        print(
            f" exported: {exported_total}, QA_DATA: {len(qa_data)}",
            file=sys.stderr,
        )

    overview_path = DATASET_DIR / "topics_overview.json"
    with overview_path.open("w", encoding="utf-8") as fh:
        json.dump(overview, fh, ensure_ascii=False, indent=2)

    print(f"Exported {len(overview)} topic files to {DATASET_DIR}")
    print(f"Overview: {overview_path}")


if __name__ == "__main__":
    main()
