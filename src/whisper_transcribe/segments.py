"""Editable, timestamped transcript stored next to the minutes.

`<name>_segments.json` keeps Whisper's segments with their audio
timings so a person can review them against the recording, fix the
text, and rebuild the minutes from the corrected version.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .core import Segment

SUFFIX = "_segments.json"
FORMAT_VERSION = 1


@dataclass
class EditableSegment:
    start: float
    end: float
    text: str
    original: str = ""
    reviewed: bool = False

    def __post_init__(self):
        if not self.original:
            self.original = self.text

    @property
    def edited(self) -> bool:
        return self.text != self.original


@dataclass
class TranscriptDoc:
    audio: str
    language: Optional[str] = None
    segments: list[EditableSegment] = field(default_factory=list)

    @classmethod
    def from_segments(cls, audio: Path, segments: list[Segment], language: Optional[str]) -> "TranscriptDoc":
        return cls(
            audio=str(audio.resolve()),
            language=language,
            segments=[EditableSegment(s.start, s.end, s.text) for s in segments],
        )

    @classmethod
    def load(cls, path: Path) -> "TranscriptDoc":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            audio=data["audio"],
            language=data.get("language"),
            segments=[EditableSegment(**s) for s in data["segments"]],
        )

    def save(self, path: Path) -> None:
        data = {"version": FORMAT_VERSION, **asdict(self)}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)  # never leave a half-written file behind

    def to_segments(self) -> list[Segment]:
        return [Segment(s.start, s.end, s.text.strip(), self.language) for s in self.segments if s.text.strip()]

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def audio_path(self, doc_path: Path) -> Path:
        """The recording, falling back to the JSON's folder if it was moved."""
        audio = Path(self.audio)
        if audio.is_file():
            return audio
        return doc_path.with_name(audio.name)


def is_segments_file(path: Path) -> bool:
    return path.name.endswith(SUFFIX)


def stem_of(path: Path) -> str:
    """Base name shared by a recording and the files generated from it."""
    return path.name[: -len(SUFFIX)] if is_segments_file(path) else path.stem


def path_for(stem: str, out_dir: Path) -> Path:
    return out_dir / f"{stem}{SUFFIX}"
