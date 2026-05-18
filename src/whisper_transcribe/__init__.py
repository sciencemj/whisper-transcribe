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
        help="Spoken language code (e.g. 'en', 'ko'). Default: auto-detect.",
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

    print(f"transcribing {args.audio.name}...", file=sys.stderr)
    result = model.transcribe(
        str(args.audio),
        language=args.language,
        verbose=True if args.per_seg else False,
    )

    output = Path.cwd() / f"{args.audio.stem}.txt"
    output.write_text(result["text"].strip() + "\n", encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)
