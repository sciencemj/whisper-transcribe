# whisper-transcribe

A small CLI that transcribes audio files (m4a, mp3, wav, …) to `.txt` using [OpenAI Whisper](https://github.com/openai/whisper), with optional GPU acceleration (CUDA / Apple Silicon MPS).

It also turns **Korean meeting recordings into meeting minutes (회의록)** — a timestamped transcript plus an AI-organized summary (agenda, discussion, decisions, action items) — with a desktop GUI. Everything runs **fully on your machine**: Whisper for speech-to-text and a local LLM server ([Ollama](https://ollama.com) or any OpenAI-compatible server such as LM Studio / llama.cpp) for the summary.

## Requirements

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- `ffmpeg` (e.g. `brew install ffmpeg` on macOS)

## Install

```bash
git clone https://github.com/sciencemj/whisper-transcribe.git
uv tool install --editable ./whisper-transcribe
```

This puts `whisper-transcribe` and `meeting-minutes` (the GUI) on your `PATH` (via `~/.local/bin`).

For AI-organized minutes, install [Ollama](https://ollama.com/download) and pull a Korean-capable model once:

```bash
ollama pull exaone3.5:7.8b    # default; LG AI Research, strong Korean (≈4.8 GB)
ollama pull exaone3.5:2.4b    # lighter option for laptops (≈1.6 GB)
```

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
| `--multi-lang` | off | Per-segment language detection for audio that switches languages mid-recording. Uses VAD to split, runs Whisper's `detect_language` per chunk, then transcribes each chunk in its detected language. Slower than the single-pass mode and quality near chunk boundaries may degrade. |
| `--format {plain,tagged,timestamped}` | `plain` | Output format for `--multi-lang`. `plain` concatenates text. `tagged` prefixes each chunk with `[lang]`. `timestamped` prefixes with `[mm:ss.ss-mm:ss.ss lang]`. |

### Examples

```bash
whisper-transcribe interview.mp3 --language en
whisper-transcribe lecture.m4a --model small --device cpu
whisper-transcribe podcast.m4a -s                          # per-segment lines instead of bar
whisper-transcribe bilingual.m4a --multi-lang              # auto language per chunk, plain text
whisper-transcribe bilingual.m4a --multi-lang --format timestamped
```

## Meeting minutes (회의록)

```bash
whisper-transcribe meeting.m4a --minutes                  # transcript-only minutes
whisper-transcribe meeting.m4a --summarize                # + local LLM summary
whisper-transcribe meeting.m4a --summarize --title "주간 회의" --attendees "김철수, 이영희" -o ~/Documents/minutes
whisper-transcribe meeting_녹취록.txt --summarize --llm-model exaone3.5:2.4b   # re-summarize a saved transcript
```

Writes `<name>_회의록.md` (header, AI summary, collapsible full transcript) and `<name>_녹취록.txt` (timestamped transcript). The transcript is saved *before* summarizing, so a failed or interrupted LLM step never loses it.

Meeting mode defaults to Korean and tunes Whisper for long recordings: a Korean punctuation prompt, `condition_on_previous_text=False` to avoid repetition loops, and filtering of common Korean hallucinations on silence (e.g. "시청해주셔서 감사합니다"). Long transcripts are summarized in parts first, then merged.

| Flag | Default | Notes |
|---|---|---|
| `--minutes` | off | Meeting mode: write `<name>_회의록.md`. `--language` defaults to `ko`. |
| `--summarize` | off | Implies `--minutes`; organize the minutes with a local LLM. Also accepts a `.txt`/`.md` transcript as input. |
| `--title`, `--attendees` | – | Shown in the minutes header. |
| `-o`, `--output-dir` | cwd | Output directory (also applies to plain transcription). |
| `--llm-backend {ollama,openai}` | `ollama` | `openai` = any OpenAI-compatible local server (LM Studio, llama.cpp `llama-server`, vLLM). |
| `--llm-model` | `exaone3.5:7.8b` | Local model name. |
| `--llm-url` | `http://localhost:11434` / `http://localhost:1234/v1` | Server URL. |
| `--num-ctx` | `8192` | Context window passed to Ollama. |
| `--chunk-chars` | `6000` | Transcripts longer than this are summarized in parts. |
| `--allow-remote-llm` | off | Permit a non-localhost LLM URL or an Ollama cloud model. |

### GUI

```bash
meeting-minutes                 # or: whisper-transcribe --gui
```

A [Flet](https://flet.dev) desktop app: pick a recording (or transcript), set title/attendees, Whisper model/language/device and the local LLM, then generate. Shows progress and logs, renders the finished minutes, and can copy or open the file. On Linux the file picker needs `zenity`.

### Staying fully local

- **This app** only talks to loopback LLM URLs and refuses Ollama *cloud* models (e.g. `gpt-oss:20b-cloud`, which forward prompts to ollama.com through the same local API) unless `--allow-remote-llm` is given. In the GUI, cloud models are hidden from the model list.
- **Ollama**: inference is local, but Ollama also offers cloud models and web search. Start it with `OLLAMA_NO_CLOUD=1 ollama serve` to disable those server-side. The network is only needed for `ollama pull`.
- **One-time downloads**: Whisper weights (`~/.cache/whisper/`), the LLM model (`ollama pull`) and the Flet desktop client (`~/.flet/client/`, on first GUI launch). After that everything works offline — the most reliable check is to simply disconnect from the network, or block outbound traffic for `ollama` with a firewall such as LuLu / Little Snitch.

## Notes

- First use of each model size downloads weights to `~/.cache/whisper/` (turbo ≈ 1.5 GB).
- On Apple Silicon, the script sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so any Whisper ops not yet implemented on MPS silently fall back to CPU.
