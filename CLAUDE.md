# CLAUDE.md

## Project Context

Always read `~/tools/AGENTS.md` first for ecosystem-wide context and development rules. Follow `~/tools/SPEC.md` for CLI design patterns.

## Overview

vidflow is a unified video capture and transcription CLI. It consolidates the former ytcapture (YouTube/local video frame extraction) and vidscribe (Claude Vision transcription) into a single installable package with four entry points: `vidflow`, `ytcapture`, `vidcapture`, and `vidscribe`.

## Architecture

### Entry points

- `vidflow youtube <url>...` — Capture YouTube video frames
- `vidflow local <file>...` — Capture local video frames
- `vidflow transcribe <markdown>...` — Transcribe captured frames
- `ytcapture` — Standalone backward-compatible YouTube capture
- `vidcapture` — Standalone backward-compatible local video capture
- `vidscribe` — Standalone backward-compatible transcription

The `--transcribe` flag on `youtube` and `local` chains capture and transcription in one step.

### Transcript handling

The transcribe module natively handles pre-existing transcript text (e.g., YouTube auto-captions) via the `existing_text` field on `TimestampSection`. When `parse_vidcapture_markdown` encounters text after image embeds, it captures it into `existing_text`. The unified prompt and template builder include `<existing-transcript>` XML tags per section when this text is present, instructing Claude to enhance/correct it using visual frame context.

Both YouTube captures (with existing transcripts) and local captures (skeleton sections) flow through the same `VidscribeProcessor`.

## Source layout

```
src/vidflow/
├── __init__.py              # __version__
├── cli.py                   # Unified vidflow entry point (argparse)
├── cli_common.py            # ExitCode, OperationResult
├── completion.py            # vidflow bash completion handler
├── youtube.py               # YouTube-specific transcription wrapper
├── data/
│   └── completion.bash      # vidflow completion script
├── capture/                 # Frame extraction (formerly ytcapture)
│   ├── __init__.py          # Public API + OperationResult wrappers
│   ├── cli.py               # Standalone ytcapture/vidcapture entry points
│   ├── completion.py        # Capture completion handler
│   ├── config.py            # ~/.config/vidflow/config.yml
│   ├── core.py              # process_video(), process_local_video()
│   ├── frames.py            # ffmpeg frame extraction
│   ├── local.py             # Local video metadata (ffprobe)
│   ├── markdown.py          # Obsidian markdown generation
│   ├── metadata.py          # VideoMetadataProtocol
│   ├── titling.py           # AI title generation (Claude Haiku)
│   ├── transcript.py        # YouTube transcript fetching
│   ├── utils.py             # URL parsing, formatting
│   ├── video.py             # yt-dlp wrapper
│   └── data/                # Bash completion scripts
└── transcribe/              # Transcription (formerly vidscribe)
    ├── __init__.py           # Public API + transcribe_markdown()
    ├── cli.py                # Standalone vidscribe entry point
    ├── models.py             # TimestampSection, VidcaptureDocument
    ├── parser.py             # Markdown parsing, merge, resolve
    ├── processor.py          # VidscribeProcessor
    ├── prompts.py            # Prompt constants, API config
    ├── image.py              # ImageMagick operations
    └── output.py             # Output path, sanitize, context loading
tests/
├── test_cli.py              # CLI entry point integration tests
├── test_clipboard.py        # URL extraction and clipboard tests
└── test_titling.py          # AI title generation tests
```

## Development

```bash
cd ~/tools/vidflow
uv sync                        # Install all dependencies
uv run vidflow --help           # Run CLI
uv run pytest tests/ -v         # Run tests
uv run black src/ tests/        # Format
uv run ruff check src/ tests/   # Lint
```

## Environment

- `ANTHROPIC_API_KEY` — Required for transcription and AI title generation
- `EXA_API_KEY` — Optional, enables citation search during transcription

## SPEC.md compliance

- ExitCode enum: SUCCESS=0, ERROR=1, USAGE_ERROR=2
- OperationResult for all operations, `--json` routes to stdout
- Human messages to stderr, JSON to stdout
- Error aggregation for multi-input processing
- TTY-aware defaults where applicable
