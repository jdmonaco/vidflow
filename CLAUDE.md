# CLAUDE.md

## Project Context

Always read `~/tools/AGENTS.md` first for ecosystem-wide context and development rules. Follow `~/tools/SPEC.md` for CLI design patterns.

## Overview

vidflow is a unified video capture and transcription CLI. It consolidates the former ytcapture (YouTube/local video frame extraction) and vidscribe (AI vision transcription) into a single installable package with one entry point, `vidflow` (the standalone `ytcapture`/`vidcapture`/`vidscribe` commands were removed in 0.6.0). Inference runs on the local ampere-gateway by default (via `~/tools/aikit`); claude-* model ids route to the Anthropic API as the escape hatch.

## Architecture

### Entry points

- `vidflow youtube <url>...` — Capture YouTube video frames
- `vidflow local <file>...` — Capture local video frames
- `vidflow transcribe <markdown>...` — Full visual transcription of captured frames
- `vidflow polish <markdown>...` — Text-only cleanup of captured caption text
- `vidflow completion bash [--install | --path]` — Bash completion (handled before argparse)

The `--transcribe` and `--polish` flags on `youtube` and `local` (mutually exclusive) chain capture and post-processing in one step.

### Post-processing tiers

Both tiers run through `VidscribeProcessor`; polish is its `text_only` mode:

- **transcribe** — frames + captions to a vision model; describes slide content and enhances/corrects caption text with visual context (`TEMPLATE_FILL_PROMPT`). One transcript per input; `--merge` combines inputs into one (H1 per source file). Generated frontmatter is merged over the input's via `merge_frontmatter`, so capture keys like `source`/`published`/`author` survive.
- **polish** — captions only, no frames sent; corrects speech-to-text errors, filler words, punctuation, and paragraphing (`POLISH_PROMPT`). Cheap and fast; requires caption text (YouTube auto-captions, or a local sidecar .vtt or embedded subtitles) in the capture. Citation search (Exa) is disabled in this mode. Polish only improves its inputs: each input is polished on its own and **in place** (frontmatter/title/preamble preserved verbatim); it never merges, retitles, or generates frontmatter, and has no `-t` or `--merge`. `-o` only redirects the polished copy (`polish_target`: a file for one input, else a directory receiving each input's own filename).

Whenever a new transcript file is written (transcribe, or polish with `-o` to another path), each input capture note is moved into the sibling `transcripts/` folder (`archive_capture`, numeric suffix on collision) so the folder holds one note per video; `--keep-capture` leaves it in place. The generated frontmatter is repaired (`repair_yaml_scalars`) when the model emits bare colons or flow openers in scalar values, and a failed generation falls back to the capture note's own title rather than a generic one.

### Transcript handling

The transcribe module natively handles pre-existing transcript text (e.g., YouTube auto-captions) via the `existing_text` field on `TimestampSection`. When `parse_vidcapture_markdown` encounters text after image embeds, it captures it into `existing_text`. The unified prompt and template builder include `<existing-transcript>` XML tags per section when this text is present, instructing Claude to enhance/correct it using visual frame context.

Local captures take caption text from a sidecar WebVTT (`find_sidecar_vtt`: `<stem>.vtt`, then `<stem>.<lang>.vtt`/`<stem>-<lang>.vtt` preferring English — Teams names downloads `<stem>-en-US.vtt`) ahead of embedded tracks; an explicit `--subtitle-track` skips the sidecar. WebVTT voice tags (`<v Name>`) set `TranscriptSegment.speaker`, and `format_segments` renders speaker turns as `**Name**: text` paragraphs — the same label format the transcribe prompt emits, and both `TEMPLATE_FILL_PROMPT` and `POLISH_PROMPT` treat existing labels as authoritative. Teams downloads of locally transcribed recordings (Word/Stream "Transcribe") carry no voice tags; only meeting transcripts are diarized.

Both YouTube captures (with existing transcripts) and local captures (captioned or skeleton sections) flow through the same `VidscribeProcessor`.

## Source layout

```
src/vidflow/
├── __init__.py              # __version__
├── cli.py                   # Unified vidflow entry point (argparse)
├── cli_common.py            # ExitCode, OperationResult
├── completion.py            # completion bash [--install | --path] handler
├── youtube.py               # YouTube-specific transcription wrapper
├── data/
│   └── completion.bash      # vidflow completion script (option lists synced to build_parser() by tests/test_completion.py)
├── capture/                 # Frame extraction (formerly ytcapture)
│   ├── __init__.py          # Public API + OperationResult wrappers
│   ├── config.py            # ~/.config/vidflow/config.yml
│   ├── core.py              # process_video(), process_local_video()
│   ├── frames.py            # ffmpeg frame extraction
│   ├── local.py             # Local video metadata (ffprobe)
│   ├── markdown.py          # Obsidian markdown generation
│   ├── metadata.py          # VideoMetadataProtocol
│   ├── titling.py           # AI title generation (local quick slot)
│   ├── transcript.py        # Transcript selection (download captions, API fallback)
│   ├── subtitles.py         # Sidecar .vtt discovery, embedded subtitle extraction, WebVTT voice tags
│   ├── utils.py             # URL parsing, clipboard URLs, formatting
│   └── video.py             # yt-dlp wrapper (single-call fetch_video)
└── transcribe/              # Transcription (formerly vidscribe)
    ├── __init__.py           # Public API + transcribe_markdown()
    ├── models.py             # TimestampSection, VidcaptureDocument
    ├── parser.py             # Markdown parsing, merge, resolve
    ├── processor.py          # VidscribeProcessor
    ├── prompts.py            # Prompt constants, API config
    ├── image.py              # ImageMagick operations
    └── output.py             # Output path, sanitize, context loading
tests/                       # pytest suite, one module per feature area
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

- `ANTHROPIC_API_KEY` — Required only for claude-* models (the Anthropic escape hatch); the local gateway lane needs no key
- `AMPERE_GATEWAY_URL` — Optional, gateway origin override (default: `http://ampere.lan:8080`)
- `EXA_API_KEY` — Optional, enables citation search during transcription (not polish)

## SPEC.md compliance

- ExitCode enum: SUCCESS=0, ERROR=1, USAGE_ERROR=2
- OperationResult for all operations, `--json` routes to stdout
- Human messages to stderr, JSON to stdout
- Error aggregation for multi-input processing
- TTY-aware defaults where applicable
