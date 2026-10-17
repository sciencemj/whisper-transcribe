"""Audio -> transcript -> (local LLM) -> meeting-minutes markdown."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import core, llm, minutes

TRANSCRIPT_SUFFIXES = {".txt", ".md"}


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


@dataclass
class MeetingOptions:
    whisper_model: str = "turbo"
    device: Optional[str] = None
    language: str = "ko"
    title: Optional[str] = None
    attendees: Optional[str] = None
    out_dir: Path = field(default_factory=Path.cwd)
    summarize: bool = True
    llm_client: Optional[llm.LocalLLM] = None
    chunk_chars: int = 6000


def run_meeting(
    source: Path,
    opts: MeetingOptions,
    log: Callable[[str], None] = _log_stderr,
    progress: Optional[Callable[[float], None]] = None,
    model=None,
) -> Path:
    """Write `<stem>_회의록.md` into `opts.out_dir` and return its path.

    `source` may be an audio file, or a previously saved transcript
    (.txt/.md) to only (re)run the LLM summary. Pass a loaded Whisper
    `model` to reuse it across calls.
    """
    info = minutes.MeetingInfo(
        title=opts.title or f"{source.stem} 회의록",
        source=source.name,
        attendees=opts.attendees,
    )

    if source.suffix.lower() in TRANSCRIPT_SUFFIXES:
        log(f"reading transcript {source.name}...")
        transcript = source.read_text(encoding="utf-8")
    else:
        if model is None:
            model = core.load_model(opts.whisper_model, opts.device, log=log)
        log(f"transcribing {source.name}...")
        segments = core.transcribe(
            model,
            source,
            language=opts.language or None,
            initial_prompt=core.KO_MEETING_PROMPT if opts.language == "ko" else None,
            filter_hallucinations=opts.language == "ko",
            # Meetings are long; carrying context makes Whisper loop on repeats.
            condition_on_previous_text=False,
            progress=progress,
        )
        if not segments:
            log("warning: no speech recognized")
        info.duration = segments[-1].end if segments else 0.0
        transcript = minutes.transcript_lines(segments)

    opts.out_dir.mkdir(parents=True, exist_ok=True)
    out = minutes.output_path(source, opts.out_dir)
    # Save the transcript first so a failed/aborted summary loses nothing.
    out.write_text(minutes.render(info, transcript), encoding="utf-8")
    raw = out.with_name(f"{source.stem}_녹취록.txt")
    if raw != source:
        raw.write_text(transcript + "\n", encoding="utf-8")
    log(f"wrote {out}")

    if opts.summarize and transcript.strip():
        client = opts.llm_client or llm.LocalLLM()
        summary = llm.summarize(client, transcript, chunk_chars=opts.chunk_chars, log=log)
        out.write_text(minutes.render(info, transcript, summary), encoding="utf-8")
        log(f"wrote {out} (with AI summary)")

    return out
