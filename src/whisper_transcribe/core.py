"""Whisper transcription core shared by the CLI and the GUI."""

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ProgressCb = Callable[[float], None]
LogCb = Callable[[str], None]

# Prompt that nudges Whisper toward punctuated, meeting-style Korean.
KO_MEETING_PROMPT = "다음은 한국어 회의 녹음입니다. 문장 부호를 정확히 사용합니다."

# Phrases Whisper tends to hallucinate on silence/noise in Korean audio
# (it was trained on lots of YouTube subtitles).
_KO_HALLUCINATIONS = [
    r"시청(해\s*)?주셔서\s*감사합니다",
    r"구독(과|\s*및)?\s*좋아요",
    r"좋아요(와|\s*및)?\s*구독",
    r"MBC\s*뉴스",
    r"KBS\s*뉴스",
    r"자막\s*(제공|by)",
    r"다음\s*영상에서\s*만나요",
]
_HALLUCINATION_RE = re.compile("|".join(_KO_HALLUCINATIONS))


@dataclass
class Segment:
    start: float
    end: float
    text: str
    lang: Optional[str] = None


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def pick_device(device: Optional[str] = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: Optional[str] = None, log: LogCb = _log_stderr):
    device = pick_device(device)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import whisper

    log(f"loading model '{name}' on {device}...")
    return whisper.load_model(name, device=device)


@contextmanager
def _patched_tqdm(progress: Optional[ProgressCb]):
    """Route whisper's internal tqdm progress to `progress(fraction)`."""
    if progress is None:
        yield
        return

    import tqdm as real_tqdm
    import whisper  # noqa: F401

    # `whisper.transcribe` is shadowed by the function of the same name.
    wt = sys.modules["whisper.transcribe"]

    class _Bar(real_tqdm.tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False
            kwargs["file"] = open(os.devnull, "w")
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            if self.total:
                progress(min(self.n / self.total, 1.0))

        def close(self):
            super().close()
            self.fp.close()

    class _Shim:
        tqdm = _Bar

    original = wt.tqdm
    wt.tqdm = _Shim
    try:
        yield
    finally:
        wt.tqdm = original


def is_hallucination(text: str) -> bool:
    return bool(_HALLUCINATION_RE.search(text))


def transcribe(
    model,
    audio_path: Path,
    language: Optional[str] = None,
    per_seg: bool = False,
    initial_prompt: Optional[str] = None,
    filter_hallucinations: bool = False,
    condition_on_previous_text: bool = True,
    progress: Optional[ProgressCb] = None,
) -> list[Segment]:
    with _patched_tqdm(progress):
        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=True if per_seg else False,
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        if filter_hallucinations and is_hallucination(text):
            continue
        segments.append(
            Segment(float(seg["start"]), float(seg["end"]), text, result.get("language"))
        )
    return segments


def transcribe_multi_lang(
    model,
    audio_path: Path,
    progress: Optional[ProgressCb] = None,
    log: LogCb = _log_stderr,
) -> list[Segment]:
    import torch
    import whisper
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio = whisper.load_audio(str(audio_path))
    audio_t = torch.from_numpy(audio)

    log("running VAD to find speech segments...")
    vad_model = load_silero_vad()
    speech = get_speech_timestamps(
        audio_t,
        vad_model,
        sampling_rate=16000,
        return_seconds=True,
    )
    if not speech:
        log("warning: no speech detected")
        return []

    n_mels = model.dims.n_mels
    segments = []
    for i, seg in enumerate(speech, 1):
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        chunk = audio[int(start_s * 16000) : int(end_s * 16000)]

        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(chunk), n_mels=n_mels
        ).to(model.device)
        _, probs = model.detect_language(mel)
        lang = max(probs, key=probs.get)

        result = model.transcribe(chunk, language=lang, verbose=False)
        text = result["text"].strip()

        preview = text if len(text) <= 60 else text[:57] + "..."
        log(f"[{i}/{len(speech)}] {fmt_time(start_s)}-{fmt_time(end_s)} {lang}: {preview}")
        if progress:
            progress(i / len(speech))

        segments.append(Segment(start_s, end_s, text, lang))
    return segments


def fmt_time(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def fmt_hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def format_segments(segments: list[Segment], fmt: str) -> str:
    if fmt == "timestamped":
        return "\n".join(
            f"[{fmt_time(s.start)}-{fmt_time(s.end)} {s.lang}] {s.text}" for s in segments
        )
    if fmt == "tagged":
        return "\n".join(f"[{s.lang}] {s.text}" for s in segments)
    return " ".join(s.text for s in segments)
