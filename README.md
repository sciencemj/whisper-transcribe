# whisper-transcribe

A small CLI that transcribes audio files (m4a, mp3, wav, …) to `.txt` using [OpenAI Whisper](https://github.com/openai/whisper), with optional GPU acceleration (CUDA / Apple Silicon MPS).

## Requirements

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- `ffmpeg` (e.g. `brew install ffmpeg` on macOS)

## Install

```bash
git clone https://github.com/sciencemj/whisper-transcribe.git
uv tool install --editable ./whisper-transcribe
```

This puts `whisper-transcribe` on your `PATH` (via `~/.local/bin`).

## Usage

```bash
whisper-transcribe audio.m4a
```

Output is written to `./<basename>.txt` in the current working directory.

### Options

| Flag | Default | Notes |
|---|---|---|
| `--model {tiny,base,small,medium,large,turbo}` | `turbo` | Whisper model size. Weights are downloaded on first use. |
| `--language CODE` | auto-detect | Spoken language code, e.g. `en`, `ko`. |
| `--device {cuda,mps,cpu}` | auto | Picks `cuda > mps > cpu` if not specified. |
| `-s`, `--per-seg` | off | Print each segment as it completes. Default is a percentage progress bar over audio length. |

### Examples

```bash
whisper-transcribe interview.mp3 --language en
whisper-transcribe lecture.m4a --model small --device cpu
whisper-transcribe podcast.m4a -s            # per-segment lines instead of bar
```

## Notes

- First use of each model size downloads weights to `~/.cache/whisper/` (turbo ≈ 1.5 GB).
- On Apple Silicon, the script sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so any Whisper ops not yet implemented on MPS silently fall back to CPU.
