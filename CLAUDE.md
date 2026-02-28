# CLAUDE.md

## Project Context

Always read `~/tools/AGENTS.md` first for ecosystem-wide context and development rules. Follow `~/tools/SPEC.md` for CLI design patterns.

## Overview

vidflow is an umbrella CLI that unifies the video processing pipeline: ytcapture (YouTube/local video frame extraction) and vidscribe (Claude Vision transcription). It bridges a key gap — YouTube videos captured by ytcapture include raw YouTube transcripts alongside extracted frames, but vidscribe expects empty skeleton sections (designed for vidcapture output). vidflow enables end-to-end workflows from video source to transcribed Obsidian note.

## Architecture

Flat subcommand structure:

- `vidflow youtube <url>...` — Capture YouTube video frames (wraps ytcapture)
- `vidflow local <file>...` — Capture local video frames (wraps vidcapture)
- `vidflow transcribe <markdown>...` — Transcribe captured frames (wraps vidscribe)

The `--transcribe` flag on `youtube` and `local` chains capture → transcription in one step.

### The YouTube→vidscribe gap

ytcapture output has per-frame YouTube transcript text already present. vidscribe's parser (`parse_vidcapture_markdown`) discards text after image embeds. vidflow's `youtube.py` module provides:

1. **Transcript-preserving parser** — extends vidscribe's parsing to keep existing transcript text
2. **Modified template construction** — includes `<existing-transcript>` XML tags per section
3. **Adapted prompt** (`YOUTUBE_TEMPLATE_FILL_PROMPT`) — instructs Claude to enhance/correct existing transcripts using visual frame context

### Integration strategy

vidflow imports ytcapture and vidscribe as library dependencies:
- `ytcapture.cli.process_video()` / `process_local_video()` for capture
- `vidscribe.VidscribeProcessor` for transcription orchestration
- Custom YouTube processing path for transcript-aware pipeline

## Source layout

```
src/vidflow/
├── __init__.py        # __version__ only
├── cli.py             # argparse entry point, subcommand dispatch
├── cli_common.py      # ExitCode, OperationResult (from SPEC.md)
├── capture.py         # Wrappers around ytcapture APIs
├── transcribe.py      # Wrapper around vidscribe VidscribeProcessor
└── youtube.py         # YouTube transcript-aware parsing + adapted prompt
tests/
└── __init__.py
```

## Development

```bash
cd ~/tools/vidflow
uv sync                        # Install deps (resolves sibling projects)
uv run vidflow --help           # Run CLI
uv run pytest                   # Run tests
uv run black src/ tests/        # Format
uv run ruff check src/ tests/   # Lint
```

## Environment

- `ANTHROPIC_API_KEY` — Required for transcription (`transcribe` subcommand and `--transcribe` flag)
- `EXA_API_KEY` — Optional, enables citation search during transcription

## SPEC.md compliance

- ExitCode enum: SUCCESS=0, ERROR=1, USAGE_ERROR=2
- OperationResult for all operations, `--json` routes to stdout
- Human messages to stderr, JSON to stdout
- Error aggregation for multi-input processing
- TTY-aware defaults where applicable
