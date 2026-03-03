"""Prompt constants and API configuration for transcription."""

# API request size limit (conservative, below 32MB API maximum)
MAX_REQUEST_SIZE_MB = 30
MAX_REQUEST_SIZE_BYTES = MAX_REQUEST_SIZE_MB * 1024 * 1024

# Default maximum image dimension (Vision API optimal)
DEFAULT_MAX_DIMENSION = 1568

# Supported image formats
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Maximum tool calls per batch (safety limit for Exa citation search)
MAX_TOOL_CALLS_PER_BATCH = 20

# Exa search tool schema for Claude Messages API
EXA_SEARCH_TOOL = {
    "name": "exa_search",
    "description": (
        "Search for academic papers and research publications. "
        "Returns title, URL, author, date, and a brief excerpt. "
        "Use this to resolve partial references seen on slides into full citations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query for finding the paper. Include author names, "
                    "year, and key title words for best results."
                ),
            }
        },
        "required": ["query"],
    },
}

# Citation search prompt appended when Exa is enabled
CITATION_SEARCH_PROMPT = """
<citation-search>
While transcribing each frame, watch for references to published papers or research works.
These may appear as:
- Author-year citations (e.g., "Smith et al., 2023")
- Partial or abbreviated paper titles on slides
- DOI strings or journal references
- Reference list slides or bibliography sections
- Named methods, tools, or datasets attributed to specific papers

For each distinct reference you detect:
1. Construct a search query combining author names, year, and key title words
2. Call the `exa_search` tool to look up the full citation
3. Use the returned information to produce an accurate citation

After the visual content and speaker text for each section, if any references were detected
in that frame, add a `### References` subsection with APA-style citations:

### References
- Author, A. B., & Author, C. D. (Year). Title of the paper. *Journal Name*, volume(issue), pages. URL

Guidelines:
- Only add references actually visible or mentioned in the frame content
- Deduplicate: if a reference appeared in a previous frame, do not re-cite it
- If a search returns no useful result, include the reference as-is with "[citation not found]"
- Do not fabricate citation details — use only what the search returns or what is visible on screen
</citation-search>
"""

# Template-fill prompt for markdown mode
TEMPLATE_FILL_PROMPT = """
<task>
You are filling in a transcript template for a video recording. Each timestamp section
contains a corresponding video frame image and may also include pre-existing transcript text
(e.g., YouTube auto-captions). Your job is to produce polished transcript content that
combines visual analysis with any existing text.
</task>

<instructions>
For each timestamp section in the template, produce TWO clearly separated output blocks:

### 1. Visual Content (slide/presentation description)

Describe ONLY relevant presented visual content: slide bullet points, titles, graph and plot
descriptions, diagrams, equations, tables, or other substantive displayed material. Write in
concise narrative form — key points and what figures convey, not a visual inventory.

- Do NOT describe faces, people, webcam feeds, or video call participant thumbnails.
- Do NOT describe screen chrome, window borders, recording indicators, or UI elements.
- If the frame shows nothing of visual interest (e.g., only a speaker's face or a blank
  screen), write "[No slide content visible]" and move on.

### 2. Speaker Text

Use ONE of the following strategies, depending on what is available for each section:

**A. Existing transcript provided** — If the section includes `<existing-transcript>` tags,
use that text as a starting point and enhance it:

- Fix obvious speech-to-text errors using surrounding context and domain knowledge.
- Clean up filler words (um, uh, like, you know) and false starts.
- Merge sentence fragments into complete, flowing sentences.
- Add proper punctuation, capitalization, and paragraph breaks.
- Remove repeated words or stuttered phrases.
- Correct technical terminology that was misrecognized by auto-captioning.
- Preserve the speaker's meaning and technical terminology faithfully.

**B. Live transcription sidebar visible** — If no existing transcript is provided but a live
transcription sidebar is visible in the frame (e.g., Teams real-time transcript panel), perform
a high-fidelity re-transcription of all visible text segments. Apply the same corrections as
strategy A.

**C. No transcript source** — If neither an existing transcript nor a transcript sidebar is
visible, omit the Speaker Text section entirely.

When multiple speakers are identifiable (from visual cues, transcript context, or existing
text), provide each speaker's text in a separate paragraph, starting with the speaker's name
in bold:

**Speaker Name**: Their cleaned-up speech content continues here as a flowing paragraph...

**Other Speaker**: Their response follows in a new paragraph...

### 3. Speaker Identification (Zoom and other platforms without live transcription)

For Zoom recordings or other platforms where no live transcript sidebar is available:

- If a gallery of attendee video tiles is visible at the top or side of the screen, identify
  the current speaker by the green border (or other highlight) around their video tile.
- Report the speaker's name as shown on their tile label: `**Speaker Name** (active speaker)`
- If no attendee has a highlighted border, note `[No active speaker indicated]`.
- Do NOT describe attendees' faces or appearances — only extract the name from the tile label.
- Combine this with any visible slide content to provide context for who is presenting.

### Continuity

The output for each frame must flow coherently and continuously from the previous frame's
content provided in context. Do not repeat content already transcribed. Pick up where the
previous frame left off — continue mid-sentence if the prior context ended mid-thought.
When the same speaker continues across frames, do not re-introduce them; just continue
their text. Only start a new speaker paragraph when the speaker changes.
</instructions>

<output-format>
Output the completed template sections. Keep exact timestamp headings and image embeds.
After each image embed, on a new line, add the visual content description (if any), then
a blank line, then the enhanced speaker text (if any). Use this structure:

## HH:MM:SS
![[image_embed]]

[Visual content description here, or "[No slide content visible]"]

**Speaker Name**: Enhanced transcript text here...
</output-format>
"""

# Frontmatter generation prompt
FRONTMATTER_PROMPT = """
<task>
Generate YAML frontmatter for a transcribed workshop/meeting recording.
</task>

<instructions>
Based on the transcript content and any background context provided, generate:

1. **title**: An informative, readable title (5-10 words) capturing the main topic
2. **tags**: 3-6 single-word tags (lowercase, no spaces) for categorization
3. **created**: Today's date (provided below)
4. **description**: 1-2 sentence summary of the content

Use the background context to inform terminology, speaker names, and organizational context.
</instructions>

<output-format>
Return ONLY valid YAML (no markdown code fences), e.g.:
title: Workshop on Neural Data Integration Methods
created: 2026-01-18
tags:
  - neuroscience
  - data
  - workshop
  - bican
description: BICAN workshop session covering neural data integration approaches and tooling.
</output-format>
"""
