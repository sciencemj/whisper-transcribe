import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file (m4a, mp3, wav, ...) to a .txt file using OpenAI Whisper.",
    )
    parser.add_argument("audio", type=Path, help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--model",
        default="turbo",
        help="Whisper model size: tiny, base, small, medium, large, turbo. Default: turbo.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Spoken language code (e.g. 'en', 'ko'). Default: auto-detect. Ignored with --multi-lang.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Compute device. Default: auto-pick (cuda > mps > cpu).",
    )
    parser.add_argument(
        "-s",
        "--per-seg",
        action="store_true",
        help="Print each transcribed segment as it completes. Default: show a percentage progress bar over audio length.",
    )
    parser.add_argument(
        "--multi-lang",
        action="store_true",
        help="Enable per-segment language detection (uses VAD to split audio, detects language per chunk). Slower; quality at chunk boundaries may degrade.",
    )
    parser.add_argument(
        "--format",
        default="plain",
        choices=["plain", "tagged", "timestamped"],
        help="Output format for --multi-lang. plain: concatenated text. tagged: '[lang] text'. timestamped: '[mm:ss.ss-mm:ss.ss lang] text'. Default: plain.",
    )
    args = parser.parse_args()

    if not args.audio.is_file():
        sys.exit(f"error: audio file not found: {args.audio}")

    import torch

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import whisper

    print(f"loading model '{args.model}' on {device}...", file=sys.stderr)
    model = whisper.load_model(args.model, device=device)

    if args.multi_lang:
        if args.language:
            print("note: --language is ignored with --multi-lang", file=sys.stderr)
        text = _transcribe_multi_lang(model, args.audio, args.format)
    else:
        print(f"transcribing {args.audio.name}...", file=sys.stderr)
        result = model.transcribe(
            str(args.audio),
            language=args.language,
            verbose=True if args.per_seg else False,
        )
        text = result["text"].strip()

    output = Path.cwd() / f"{args.audio.stem}.txt"
    output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)


def _fmt_time(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


def _transcribe_multi_lang(model, audio_path: Path, fmt: str) -> str:
    import torch
    import whisper
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio = whisper.load_audio(str(audio_path))
    audio_t = torch.from_numpy(audio)

    print("running VAD to find speech segments...", file=sys.stderr)
    vad_model = load_silero_vad()
    segments = get_speech_timestamps(
        audio_t,
        vad_model,
        sampling_rate=16000,
        return_seconds=True,
    )
    if not segments:
        print("warning: no speech detected", file=sys.stderr)
        return ""

    n_mels = model.dims.n_mels
    parts = []
    for i, seg in enumerate(segments, 1):
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
        print(
            f"[{i}/{len(segments)}] {_fmt_time(start_s)}-{_fmt_time(end_s)} {lang}: {preview}",
            file=sys.stderr,
        )

        if fmt == "timestamped":
            parts.append(f"[{_fmt_time(start_s)}-{_fmt_time(end_s)} {lang}] {text}")
        elif fmt == "tagged":
            parts.append(f"[{lang}] {text}")
        else:
            parts.append(text)

    return "\n".join(parts) if fmt != "plain" else " ".join(parts)
