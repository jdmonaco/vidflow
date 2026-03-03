# vidflow

Unified video capture and transcription CLI. Consolidates YouTube/local video frame extraction (formerly ytcapture) and Claude Vision transcription (formerly vidscribe) into a single installable package.

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

# Capture + transcribe in one step
vidflow youtube URL --transcribe

# Multiple videos
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
  |
  +- vidflow.capture.core.process_video()      -> markdown with YouTube transcript
  |
  +- vidflow.youtube.transcribe_youtube()
       +- parse_vidcapture_markdown()           -> preserves existing transcript per section
       +- VidscribeProcessor.process_all()      -> Claude Vision transcription

vidflow local file.mp4 --transcribe
  |
  +- vidflow.capture.core.process_local_video() -> markdown (empty sections)
  |
  +- vidflow.transcribe.transcribe_markdown()
       +- VidscribeProcessor.process_all()      -> standard skeleton transcription
```

When transcribing YouTube captures, existing auto-caption text is passed to Claude via `<existing-transcript>` tags, instructing it to enhance and correct using visual frame context rather than transcribing from scratch.

## Multi-input behavior

| Command | Default | With `--merge` |
|---------|---------|----------------|
| `youtube URL1 URL2` | Independent (2 outputs) | Merged (1 output) |
| `local f1.mp4 f2.mp4` | Independent (2 outputs) | Merged (1 output) |
| `transcribe f1.md f2.md` | Merged (1 output) | N/A (always merged) |

## Version

0.2.1
