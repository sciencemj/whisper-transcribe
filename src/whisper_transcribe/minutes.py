"""Turn transcribed segments into a Korean meeting-minutes markdown document."""

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .core import Segment, fmt_hms


@dataclass
class MeetingInfo:
    title: str
    source: str
    date: str = field(default_factory=lambda: dt.date.today().isoformat())
    duration: Optional[float] = None
    attendees: Optional[str] = None


def group_paragraphs(
    segments: list[Segment], gap: float = 2.0, max_len: float = 60.0
) -> list[Segment]:
    """Merge Whisper's short segments into readable paragraphs.

    A new paragraph starts after a pause longer than `gap` seconds or once
    the current one spans more than `max_len` seconds.
    """
    paragraphs: list[Segment] = []
    for seg in segments:
        cur = paragraphs[-1] if paragraphs else None
        if cur and seg.start - cur.end <= gap and seg.end - cur.start <= max_len:
            cur.end = seg.end
            cur.text = f"{cur.text} {seg.text}"
        else:
            paragraphs.append(Segment(seg.start, seg.end, seg.text, seg.lang))
    return paragraphs


def transcript_lines(segments: list[Segment]) -> str:
    """Timestamped plain-text transcript, also used as LLM input."""
    return "\n".join(f"[{fmt_hms(p.start)}] {p.text}" for p in group_paragraphs(segments))


def _header(info: MeetingInfo) -> str:
    rows = [
        ("일시", info.date),
        ("원본 파일", f"`{info.source}`"),
    ]
    if info.duration is not None:
        rows.append(("녹음 길이", fmt_hms(info.duration)))
    if info.attendees:
        rows.append(("참석자", info.attendees))
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"# {info.title}\n\n| 항목 | 내용 |\n|---|---|\n{table}\n"


def render(info: MeetingInfo, transcript: str, summary: Optional[str] = None) -> str:
    parts = [_header(info)]
    if summary:
        parts.append(summary.strip() + "\n")
        parts.append("---\n\n## 전체 녹취록\n")
        parts.append(
            "<details>\n<summary>펼쳐 보기</summary>\n\n"
            + _transcript_md(transcript)
            + "\n</details>\n"
        )
    else:
        parts.append("## 녹취록\n")
        parts.append(_transcript_md(transcript))
    return "\n".join(parts)


def _transcript_md(transcript: str) -> str:
    # Blank lines between paragraphs so markdown renders each on its own.
    return "\n\n".join(line for line in transcript.splitlines() if line.strip()) + "\n"


def output_path(stem: str, out_dir: Path) -> Path:
    return out_dir / f"{stem}_회의록.md"
