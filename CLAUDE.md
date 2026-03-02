# CLAUDE.md

## Project Context

Always read `~/tools/AGENTS.md` first for ecosystem-wide context and development rules. Follow `~/tools/SPEC.md` for CLI design patterns.

## Overview

vidflow is an umbrella CLI that unifies the video processing pipeline: ytcapture (YouTube/local video frame extraction) and vidscribe (Claude Vision transcription). It provides end-to-end workflows from video source to transcribed Obsidian note.

## Architecture

Flat subcommand structure:

- `vidflow youtube <url>...` — Capture YouTube video frames (wraps ytcapture)
- `vidflow local <file>...` — Capture local video frames (wraps vidcapture)
- `vidflow transcribe <markdown>...` — Transcribe captured frames (wraps vidscribe)

The `--transcribe` flag on `youtube` and `local` chains capture → transcription in one step.

### Transcript handling

vidscribe natively handles pre-existing transcript text (e.g., YouTube auto-captions) via the `existing_text` field on `TimestampSection`. When `parse_vidcapture_markdown` encounters text after image embeds, it captures it into `existing_text`. The unified prompt and template builder include `<existing-transcript>` XML tags per section when this text is present, instructing Claude to enhance/correct it using visual frame context.

This means both YouTube captures (with existing transcripts) and local captures (skeleton sections) flow through the same vidscribe `VidscribeProcessor` — no custom parser, prompt, or subclass needed in vidflow.

### Integration strategy

vidflow imports ytcapture and vidscribe as library dependencies:
- `ytcapture.cli.process_video()` / `process_local_video()` for capture
- `vidscribe.parse_vidcapture_markdown()` for markdown parsing
- `vidscribe.VidscribeProcessor.process_all()` for transcription orchestration

## Source layout

```
src/vidflow/
├── __init__.py        # __version__ only
├── cli.py             # argparse entry point, subcommand dispatch
├── cli_common.py      # ExitCode, OperationResult (from SPEC.md)
├── capture.py         # Wrappers around ytcapture APIs
├── transcribe.py      # Wrapper around vidscribe VidscribeProcessor
└── youtube.py         # YouTube transcription (thin wrapper around vidscribe)
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
