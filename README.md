# vidflow

Unified video capture and transcription CLI. Bridges [ytcapture](../ytcapture) (YouTube/local video frame extraction) and [vidscribe](../vidscribe) (Claude Vision transcription) into a single end-to-end pipeline.

## The problem

ytcapture extracts frames from YouTube videos and includes auto-generated YouTube transcript text alongside each frame. vidscribe expects empty frame skeletons (designed for vidcapture local output) and transcribes from scratch using Claude Vision. There's no path from YouTube video → Claude-enhanced transcript that leverages both the existing YouTube text and visual frame analysis.

## The solution

vidflow provides:

- **`vidflow youtube`** — Capture YouTube video frames, optionally transcribe in one step
- **`vidflow local`** — Capture local video frames, optionally transcribe in one step
- **`vidflow transcribe`** — Transcribe previously captured frame markdown files

When transcribing YouTube captures, vidflow uses a transcript-aware pipeline that preserves existing YouTube transcript text and instructs Claude to enhance, correct, and augment it with visual context from frame images — rather than transcribing from scratch.

## Install

```bash
cd ~/tools/vidflow
uv sync
```

Requires sibling directories `../ytcapture` and `../vidscribe` (resolved via `[tool.uv.sources]`).

## Usage

### YouTube capture

```bash
# Capture frames only
vidflow youtube https://youtube.com/watch?v=VIDEO_ID

# Capture + transcribe in one step
vidflow youtube https://youtube.com/watch?v=VIDEO_ID --transcribe

# Multiple videos, independent processing
vidflow youtube URL1 URL2 --transcribe

# Multiple videos, merged into one transcript
vidflow youtube URL1 URL2 --transcribe --merge
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

### Common options

```bash
# JSON output (stdout, for piping)
vidflow youtube URL --json

# Custom model
vidflow youtube URL --transcribe -m claude-opus-4-20250514

# Background context for transcription
vidflow transcribe capture.md -c agenda.md -c speakers.md

# Override title
vidflow transcribe capture.md -t "Workshop Day 1"
```

## Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For transcription | Claude API key |
| `EXA_API_KEY` | No | Enables citation search during transcription |

## Architecture

```
vidflow youtube URL --transcribe
  │
  ├─ ytcapture.cli.process_video()     → capture markdown with YouTube transcript
  │
  └─ vidflow.youtube.transcribe_youtube()
       ├─ parse_youtube_markdown()      → preserves existing transcript per section
       ├─ build_youtube_template()      → <existing-transcript> tags in template
       └─ process_youtube_batches()     → YouTube-adapted prompt + vidscribe infrastructure

vidflow local file.mp4 --transcribe
  │
  ├─ ytcapture.cli.process_local_video()  → capture markdown (empty sections)
  │
  └─ vidflow.transcribe.transcribe_markdown()
       └─ vidscribe.VidscribeProcessor     → standard skeleton transcription

vidflow transcribe capture.md
  │
  └─ vidflow.transcribe.transcribe_markdown()
       └─ vidscribe.VidscribeProcessor     → standard skeleton transcription
```

## Multi-input behavior

| Command | Default | With `--merge` |
|---------|---------|----------------|
| `youtube URL1 URL2` | Independent (2 outputs) | Merged (1 output) |
| `local f1.mp4 f2.mp4` | Independent (2 outputs) | Merged (1 output) |
| `transcribe f1.md f2.md` | Merged (1 output) | N/A (always merged) |

## Version

0.1.0
