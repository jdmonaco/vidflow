# vidflow

Unified video capture and transcription CLI. Consolidates YouTube/local video frame extraction (formerly ytcapture) and AI vision transcription (formerly vidscribe) into a single installable package. Transcription runs on the local inference stack by default (ampere-gateway `primary` slot via `~/tools/aikit`), with claude-* models as an Anthropic-API escape hatch.

## Install

```bash
cd ~/tools/vidflow
uv sync
```

Or install as a tool:

```bash
uv tool install ~/tools/vidflow
```

This provides four commands: `vidflow`, `ytcapture`, `vidcapture`, `vidscribe`.

## Usage

### YouTube capture

```bash
# Capture frames only
vidflow youtube https://youtube.com/watch?v=VIDEO_ID

# Bare video IDs also work
ytcapture dQw4w9WgXcQ

# No arguments: YouTube URLs are read from the clipboard (macOS),
# listed, and confirmed before capture (-y skips the prompt)
vidflow youtube

# Capture + full visual transcription in one step
vidflow youtube URL --transcribe

# Capture + text-only caption polish (cheaper: no frames sent)
vidflow youtube URL --polish

# Multiple videos (always independent — one note per video)
vidflow youtube URL1 URL2 --transcribe

# Playlist URLs expand to their videos (one note per video);
# watch URLs carrying a &list= param capture just that video
vidflow youtube "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Plan a run offline: which videos would be captured or skipped (already
# in the output directory), with no downloads or model calls
vidflow youtube URL1 URL2 --polish --dry-run
```

### Local video capture

```bash
# Capture frames from local file
vidflow local recording.mp4

# Capture + transcribe
vidflow local recording.mp4 --transcribe

# Multiple files merged
vidflow local part1.mp4 part2.mp4 --transcribe --merge
```

### Transcribe existing captures

```bash
# Transcribe a single capture markdown
vidflow transcribe capture.md

# Merge multiple captures into one transcript
vidflow transcribe part1.md part2.md -o combined.md

# Estimate token usage before processing
vidflow transcribe capture.md --estimate-only

# Dry run
vidflow transcribe capture.md --dry-run
```

### Polish existing captures (text-only)

`polish` is the lightweight alternative to `transcribe`: it sends only the collated caption text (YouTube auto-captions or embedded subtitles) to the configured model for cleanup — speech-to-text errors, filler words, punctuation, paragraphing — without sending any frame images. Sections without caption text pass through unchanged; frames-only captures are rejected (use `transcribe`).

A single input is polished **in place**: the section text is replaced while the file's frontmatter, title, and preamble (video embed, description) are preserved verbatim, and no frontmatter is generated. The raw captions remain recoverable in `transcripts/raw-transcript-<id>.json`. Passing `-o`, or multiple inputs (always merged), writes a new file whose generated frontmatter is merged over the original — capture keys like `source`, `published`, and `author` are preserved.

```bash
# Polish a capture markdown in place
vidflow polish capture.md

# Write a new polished file instead
vidflow polish capture.md -o polished.md

# Merge multiple captures into one polished transcript
vidflow polish part1.md part2.md -o combined.md

# Estimate token usage before processing
vidflow polish capture.md --estimate-only
```

### Standalone commands

The backward-compatible standalone entry points work the same as before:

```bash
ytcapture URL              # YouTube capture
vidcapture meeting.mp4     # Local video capture
vidscribe capture.md       # Transcription
```

### Common options

```bash
# JSON output (stdout, for piping)
vidflow youtube URL --json

# Custom model
vidflow youtube URL --transcribe -m claude-opus-5   # Anthropic escape hatch

# Background context for transcription
vidflow transcribe capture.md -c agenda.md -c speakers.md

# Override title
vidflow transcribe capture.md -t "Workshop Day 1"
```

## Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | claude-* models only | Anthropic API key for the escape-hatch lane |
| `AMPERE_GATEWAY_URL` | Optional | Gateway origin (default: `http://ampere.lan:8080`) |
| `EXA_API_KEY` | No | Enables citation search during transcription |

## Architecture

```
vidflow youtube URL --transcribe
  |
  +- vidflow.capture.core.process_video()      -> markdown with YouTube transcript
  |
  +- vidflow.youtube.transcribe_youtube()
       +- parse_vidcapture_markdown()           -> preserves existing transcript per section
       +- VidscribeProcessor.process_all()      -> AI vision transcription (local or Anthropic)

vidflow local file.mp4 --transcribe
  |
  +- vidflow.capture.core.process_local_video() -> markdown (empty sections)
  |
  +- vidflow.transcribe.transcribe_markdown()
       +- VidscribeProcessor.process_all()      -> standard skeleton transcription
```

When transcribing YouTube captures, existing auto-caption text is passed to the model via `<existing-transcript>` tags, instructing it to enhance and correct using visual frame context rather than transcribing from scratch.

```
vidflow polish capture.md
  |
  +- vidflow.transcribe.polish_markdown()
       +- parse_vidcapture_markdown()           -> collects caption text per section
       +- VidscribeProcessor(text_only=True)    -> text-only cleanup, no frames sent
```

Polish reuses the same processor, batching, retry, and continuity machinery as transcribe; `text_only` mode swaps the prompt (`POLISH_PROMPT`), skips image preparation, and disables citation search.

## Multi-input behavior

| Command | Default | With `--merge` |
|---------|---------|----------------|
| `youtube URL1 URL2` | Independent (2 outputs) | — (no merge; one note per video) |
| `local f1.mp4 f2.mp4` | Independent (2 outputs) | Merged (1 output) |
| `transcribe f1.md f2.md` | Merged (1 output) | N/A (always merged) |
| `polish f1.md` | In-place update | — |
| `polish f1.md f2.md` | Merged (1 new output) | N/A (always merged) |

`--merge` exists for stitching one long event (e.g., a workshop recorded as several local files) into a single note. A merged output keeps each source file as its own section: an H1 heading per original file (its title), with H2 timestamp headings restarting under each. The overall generated title lives in the frontmatter only. Parts are processed in separate batches with continuity context reset at each boundary, so transcription never bleeds across recordings.

## Version

0.4.2
