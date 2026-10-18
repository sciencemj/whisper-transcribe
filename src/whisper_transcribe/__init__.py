import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file (m4a, mp3, wav, ...) to a .txt file using OpenAI Whisper, "
        "or turn a meeting recording into minutes summarized by a local LLM.",
    )
    parser.add_argument(
        "audio",
        type=Path,
        nargs="?",
        help="Path to the audio file to transcribe. Also accepts a reviewed '<name>_segments.json' "
        "(rebuilds the minutes), or with --summarize a .txt/.md transcript.",
    )
    parser.add_argument(
        "--model",
        default="turbo",
        help="Whisper model size: tiny, base, small, medium, large, turbo. Default: turbo.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Spoken language code (e.g. 'en', 'ko'). Default: auto-detect ('ko' with --minutes). Ignored with --multi-lang.",
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

    meeting = parser.add_argument_group("meeting minutes")
    meeting.add_argument(
        "--minutes",
        action="store_true",
        help="Meeting mode: write '<name>_회의록.md' with a timestamped transcript (Korean by default), "
        "plus '<name>_segments.json' for reviewing it against the audio in the GUI.",
    )
    meeting.add_argument(
        "--summarize",
        action="store_true",
        help="Implies --minutes. Also organize the minutes (agenda, decisions, action items) with a local LLM.",
    )
    meeting.add_argument("--title", default=None, help="Meeting title. Default: '<name> 회의록'.")
    meeting.add_argument("--attendees", default=None, help="Attendees, e.g. '김철수, 이영희'.")
    meeting.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Output directory. Default: current directory."
    )

    local_llm = parser.add_argument_group("local LLM (used with --summarize)")
    local_llm.add_argument(
        "--llm-backend",
        default="ollama",
        choices=["ollama", "openai"],
        help="ollama: Ollama's native API. openai: any OpenAI-compatible local server (LM Studio, llama.cpp, vLLM). Default: ollama.",
    )
    local_llm.add_argument("--llm-model", default=None, help="Local model name. Default: exaone3.5:7.8b.")
    local_llm.add_argument(
        "--llm-url",
        default=None,
        help="Server URL. Default: http://localhost:11434 (ollama) / http://localhost:1234/v1 (openai).",
    )
    local_llm.add_argument(
        "--num-ctx", type=int, default=8192, help="Context window passed to Ollama. Default: 8192."
    )
    local_llm.add_argument(
        "--chunk-chars",
        type=int,
        default=6000,
        help="Transcripts longer than this are summarized in parts first. Default: 6000.",
    )
    local_llm.add_argument(
        "--allow-remote-llm",
        action="store_true",
        help="Allow a non-localhost --llm-url. Off by default so meeting content stays on this machine.",
    )

    parser.add_argument("--gui", action="store_true", help="Launch the desktop GUI.")
    args = parser.parse_args()

    if args.gui:
        from .gui import main as gui_main

        gui_main()
        return

    if args.audio is None:
        parser.error("the following arguments are required: audio (or use --gui)")
    if not args.audio.is_file():
        sys.exit(f"error: file not found: {args.audio}")

    from .segments import is_segments_file

    if args.minutes or args.summarize or is_segments_file(args.audio):
        _run_minutes(args)
    else:
        _run_transcribe(args)


def _run_minutes(args) -> None:
    from . import llm, pipeline

    if args.audio.suffix.lower() in pipeline.TRANSCRIPT_SUFFIXES and not args.summarize:
        sys.exit("error: a transcript file as input only makes sense with --summarize")

    client = None
    if args.summarize:
        try:
            client = llm.LocalLLM(
                model=args.llm_model or llm.DEFAULT_MODEL,
                backend=args.llm_backend,
                url=args.llm_url,
                num_ctx=args.num_ctx,
                allow_remote=args.allow_remote_llm,
            )
        except llm.LLMError as e:
            sys.exit(f"error: {e}")

    opts = pipeline.MeetingOptions(
        whisper_model=args.model,
        device=args.device,
        language=args.language or "ko",
        title=args.title,
        attendees=args.attendees,
        out_dir=args.output_dir or Path.cwd(),
        summarize=args.summarize,
        llm_client=client,
        chunk_chars=args.chunk_chars,
    )
    try:
        pipeline.run_meeting(args.audio, opts)
    except llm.LLMError as e:
        sys.exit(f"error: summary failed: {e}\n(the transcript-only minutes were still saved)")


def _run_transcribe(args) -> None:
    from . import core

    model = core.load_model(args.model, args.device)

    if args.multi_lang:
        if args.language:
            print("note: --language is ignored with --multi-lang", file=sys.stderr)
        segments = core.transcribe_multi_lang(model, args.audio)
        text = core.format_segments(segments, args.format)
    else:
        print(f"transcribing {args.audio.name}...", file=sys.stderr)
        segments = core.transcribe(model, args.audio, language=args.language, per_seg=args.per_seg)
        text = core.format_segments(segments, "plain")

    out_dir = args.output_dir or Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.audio.stem}.txt"
    output.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {output}", file=sys.stderr)
