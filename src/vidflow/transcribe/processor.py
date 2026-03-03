"""VidscribeProcessor — core transcription orchestrator."""

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
import yaml
from anthropic import Anthropic
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from vidflow.transcribe.image import find_magick_command, resize_image
from vidflow.transcribe.models import TimestampSection, VidcaptureDocument
from vidflow.transcribe.prompts import (
    CITATION_SEARCH_PROMPT,
    EXA_SEARCH_TOOL,
    FRONTMATTER_PROMPT,
    MAX_REQUEST_SIZE_BYTES,
    MAX_REQUEST_SIZE_MB,
    MAX_TOOL_CALLS_PER_BATCH,
    TEMPLATE_FILL_PROMPT,
)

# Optional Exa citation search support
try:
    from exa_py import Exa as ExaClient

    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False


class VidscribeProcessor:
    """Main processor for video frame transcription."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.2,
        batch_size: int = 10,
        context_frames: int = 3,
        max_dimension: int = 1568,
        background_context: str = "",
        json_output: bool = False,
        exa_api_key: Optional[str] = None,
    ):
        """Initialize the processor with API credentials and settings."""
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.context_frames = context_frames
        self.max_dimension = max_dimension
        self.background_context = background_context
        self.json_output = json_output
        self.console = Console(quiet=json_output)
        self.magick_cmd = find_magick_command()

        # Exa citation search
        self.exa_enabled = False
        self.exa_client = None
        if exa_api_key:
            if EXA_AVAILABLE:
                self.exa_client = ExaClient(api_key=exa_api_key)
                self.exa_enabled = True
            else:
                self.console.print(
                    "[yellow]Warning: EXA_API_KEY is set but exa-py is not installed. "
                    "Install with: uv pip install 'vidflow[citations]'[/yellow]"
                )

    def estimate_tokens(self, sections: List[TimestampSection]) -> int:
        """Estimate total input tokens for a list of sections."""
        prompt_tokens = len(TEMPLATE_FILL_PROMPT) // 4
        if self.exa_enabled:
            prompt_tokens += len(CITATION_SEARCH_PROMPT) // 4
        context_tokens = (
            len(self.background_context) // 4 if self.background_context else 0
        )
        image_tokens = len(sections) * 1200  # Conservative middle estimate
        existing_text_tokens = sum(
            len(s.existing_text) // 4 for s in sections if s.existing_text
        )
        return prompt_tokens + context_tokens + image_tokens + existing_text_tokens

    def image_to_base64(self, image_path: Path) -> str:
        """Convert an image file to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def get_media_type(self, image_path: Path) -> str:
        """Get the MIME type for an image file."""
        suffix = image_path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        return media_types.get(suffix, "image/png")

    def prepare_image(self, image_path: Path, temp_dir: Path, index: int) -> Path:
        """Prepare a single image by resizing if needed.

        Returns the path to the prepared image.
        """
        prepared_path = temp_dir / f"prepared_{index:06d}{image_path.suffix}"
        resize_image(image_path, prepared_path, self.max_dimension, self.magick_cmd)
        return prepared_path

    def _make_streaming_api_request(
        self,
        messages: list,
        progress_task,
        progress,
        max_retries: int = 3,
        max_tokens: int = 16000,
        tools: Optional[list] = None,
    ) -> Tuple[str, str, object]:
        """Make streaming API request with retry logic for rate limiting.

        Returns:
            tuple: (response_text, stop_reason, final_message)
        """
        for attempt in range(max_retries):
            try:
                full_response = ""
                stream_kwargs = dict(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    messages=messages,
                )
                if tools:
                    stream_kwargs["tools"] = tools

                with self.client.messages.stream(**stream_kwargs) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        progress_pct = min(95, len(full_response) / 20)
                        progress.update(progress_task, completed=progress_pct)

                    final_message = stream.get_final_message()

                progress.update(progress_task, completed=100)
                return full_response, final_message.stop_reason, final_message

            except anthropic.RateLimitError as e:
                if attempt == max_retries - 1:
                    raise

                retry_after = getattr(e, "retry_after", None) or 60
                wait_time = retry_after * (2**attempt)

                progress.update(
                    progress_task,
                    description=f"Rate limit hit. Retrying in {wait_time}s...",
                )
                time.sleep(wait_time)
                progress.update(
                    progress_task,
                    description=f"Transcribing batch with {self.model}",
                )

            except Exception as e:
                raise RuntimeError(f"API request failed: {str(e)}")

        raise RuntimeError("All retry attempts failed")

    def _execute_exa_search(self, query: str) -> str:
        """Execute an Exa academic paper search and return formatted result."""
        try:
            results = self.exa_client.search_and_contents(
                query,
                type="auto",
                category="research paper",
                num_results=1,
                text={"max_characters": 500},
            )

            if not results.results:
                return f"No results found for: {query}"

            r = results.results[0]
            parts = []
            if r.title:
                parts.append(f"Title: {r.title}")
            if r.url:
                parts.append(f"URL: {r.url}")
            if r.author:
                parts.append(f"Author: {r.author}")
            if r.published_date:
                parts.append(f"Date: {r.published_date}")
            if r.text:
                parts.append(f"Excerpt: {r.text[:300]}")
            return "\n".join(parts) if parts else f"No details found for: {query}"

        except Exception as e:
            return f"Search failed for '{query}': {str(e)}"

    def _get_batch_prompt(self) -> str:
        """Return the prompt text for batch processing."""
        prompt_text = TEMPLATE_FILL_PROMPT
        if self.exa_enabled:
            prompt_text += CITATION_SEARCH_PROMPT
        return prompt_text

    def _build_batch_template(self, sections: List[TimestampSection]) -> str:
        """Build the template-to-fill for a batch of sections.

        Includes `<existing-transcript>` tags for sections that have
        pre-existing transcript text (e.g., YouTube auto-captions).
        """
        template_text = "\n<template-to-fill>\n"
        for sec in sections:
            template_text += f"## {sec.timestamp}\n{sec.image_embed}\n"
            if sec.existing_text:
                template_text += (
                    f"<existing-transcript>\n"
                    f"{sec.existing_text}\n"
                    f"</existing-transcript>\n"
                )
            template_text += "\n"
        template_text += "</template-to-fill>"
        return template_text

    def process_markdown_batch(
        self,
        sections: List[TimestampSection],
        previous_sections: List[TimestampSection],
        temp_dir: Path,
        progress,
        batch_num: int,
        total_batches: int,
    ) -> List[str]:
        """Process a batch of markdown sections with their images.

        Args:
            sections: Sections to process in this batch
            previous_sections: Previous sections for context
            temp_dir: Temporary directory for prepared images
            progress: Rich progress instance
            batch_num: Current batch number (1-indexed)
            total_batches: Total number of batches

        Returns:
            List of transcript content for each section
        """
        content = []

        # Images first (per Claude Vision API guidance: images before text)
        content.append({"type": "text", "text": "<frame-images>\n"})

        # Track request size (estimate text content size for limit checking)
        text_size_estimate = len(self._get_batch_prompt()) + len(self.background_context)
        request_size_bytes = text_size_estimate
        images_included = 0

        # Add images for each section
        for i, section in enumerate(sections):
            if not section.image_path.exists():
                self.console.print(
                    f"[yellow]Warning: Image not found: {section.image_path}[/yellow]"
                )
                continue

            # Prepare (resize) the image
            prepared_path = self.prepare_image(
                section.image_path, temp_dir, batch_num * 1000 + i
            )
            base64_image = self.image_to_base64(prepared_path)
            image_size_bytes = len(base64_image.encode("utf-8"))

            # Check size limit
            if request_size_bytes + image_size_bytes > MAX_REQUEST_SIZE_BYTES:
                self.console.print(
                    f"[yellow]Request size limit ({MAX_REQUEST_SIZE_MB}MB) "
                    f"reached after {images_included}/{len(sections)} "
                    f"images in batch[/yellow]"
                )
                break

            request_size_bytes += image_size_bytes
            images_included += 1

            # Add timestamp label
            content.append(
                {"type": "text", "text": f"\n[Frame: {section.timestamp}]\n"}
            )

            # Add image
            image_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": self.get_media_type(prepared_path),
                    "data": base64_image,
                },
            }
            content.append(image_block)

        # Close the frame-images tag
        content.append({"type": "text", "text": "\n</frame-images>\n\n"})

        # Add background context if available (with cache control for prompt caching)
        if self.background_context:
            content.append(
                {
                    "type": "text",
                    "text": f"<background-context>\n{self.background_context}\n</background-context>\n\n",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            )

        # Add previous transcription context if available
        if previous_sections:
            context_text = "<previous-transcription-context>\n"
            context_text += "The following are the most recent transcribed sections. Use for continuity:\n\n"
            for sec in previous_sections:
                context_text += (
                    f"## {sec.timestamp}\n{sec.image_embed}\n{sec.content}\n\n"
                )
            context_text += "</previous-transcription-context>\n\n"
            content.append({"type": "text", "text": context_text})

        # Add the template-fill prompt and template
        content.append({"type": "text", "text": self._get_batch_prompt()})
        content.append({"type": "text", "text": self._build_batch_template(sections)})

        # Make API request
        api_task = progress.add_task(
            f"Transcribing batch {batch_num}/{total_batches} with {self.model}",
            total=100,
        )

        messages = [{"role": "user", "content": content}]

        # Pass tools if Exa citation search is enabled
        tools = [EXA_SEARCH_TOOL] if self.exa_enabled else None

        response_text, stop_reason, final_message = self._make_streaming_api_request(
            messages, api_task, progress, tools=tools
        )

        # Handle tool use loop (Exa citation searches)
        tool_call_count = 0
        while (
            stop_reason == "tool_use"
            and self.exa_enabled
            and tool_call_count < MAX_TOOL_CALLS_PER_BATCH
        ):
            # Extract tool_use blocks from the response
            tool_use_blocks = [
                block
                for block in final_message.content
                if block.type == "tool_use"
            ]

            if not tool_use_blocks:
                break

            # Reconstruct assistant message with all content blocks
            assistant_content = []
            for block in final_message.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call and collect results
            tool_results = []
            for block in tool_use_blocks:
                tool_call_count += 1
                query = block.input.get("query", "")
                self.console.print(
                    f"[dim]  Searching: {query[:80]}...[/dim]"
                    if len(query) > 80
                    else f"[dim]  Searching: {query}[/dim]"
                )
                result = self._execute_exa_search(query)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

            progress.update(
                api_task,
                description=f"Processing citations ({tool_call_count} searches)",
                completed=0,
            )

            continued_text, stop_reason, final_message = (
                self._make_streaming_api_request(
                    messages, api_task, progress, tools=tools
                )
            )
            response_text += continued_text

        if tool_call_count >= MAX_TOOL_CALLS_PER_BATCH:
            self.console.print(
                f"[yellow]Warning: Hit tool call limit "
                f"({MAX_TOOL_CALLS_PER_BATCH}) for this batch[/yellow]"
            )

        # Handle continuation if truncated (multi-turn, no prefill)
        max_continuations = 3
        continuation_count = 0

        while (
            stop_reason in ["max_tokens", "pause_turn"]
            and continuation_count < max_continuations
        ):
            continuation_count += 1
            self.console.print(
                f"[yellow]Response truncated ({stop_reason}), continuing... "
                f"(attempt {continuation_count}/{max_continuations})[/yellow]"
            )

            # Serialize the API-returned content blocks as an assistant message
            assistant_content = []
            for block in final_message.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # Append a user-role continuation prompt
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response was truncated. Continue the transcript "
                    "exactly where you left off. Do not repeat any content already "
                    "provided."
                ),
            })

            progress.update(
                api_task,
                description=f"Continuing transcription ({continuation_count})",
                completed=0,
            )

            continued_text, stop_reason, final_message = (
                self._make_streaming_api_request(messages, api_task, progress)
            )
            response_text += continued_text

        if stop_reason in ["max_tokens", "pause_turn"]:
            self.console.print(
                f"[yellow]Warning: Response still truncated after "
                f"{max_continuations} continuation attempts[/yellow]"
            )

        # Remove the task to prevent accumulation in the progress display
        progress.remove_task(api_task)

        # Parse response to extract content for each section
        return self._parse_batch_response(response_text, sections)

    def _parse_batch_response(
        self, response_text: str, sections: List[TimestampSection]
    ) -> List[str]:
        """Parse the API response to extract content for each section."""
        results = []

        # Pattern to match timestamp sections in response
        section_pattern = re.compile(
            r"##\s+(\d{2}:\d{2}:\d{2})\s*\n(.*?)(?=##\s+\d{2}:\d{2}:\d{2}|\Z)",
            re.DOTALL,
        )

        # Parse response sections
        response_sections = {}
        for match in section_pattern.finditer(response_text):
            timestamp = match.group(1)
            content = match.group(2).strip()

            # Remove the image embed line if present (we'll add it ourselves)
            content = re.sub(r"!\[\[[^\]]+\]\]\s*", "", content).strip()
            response_sections[timestamp] = content

        # Match up with our sections
        for section in sections:
            if section.timestamp in response_sections:
                results.append(response_sections[section.timestamp])
            else:
                results.append("")
                self.console.print(
                    f"[yellow]Warning: No content found for {section.timestamp}[/yellow]"
                )

        return results

    def generate_frontmatter(self, transcript: str) -> Dict:
        """Generate YAML frontmatter based on transcript content."""
        # Build the prompt
        prompt_text = FRONTMATTER_PROMPT
        prompt_text += f"\n\nToday's date: {date.today().isoformat()}\n"

        if self.background_context:
            prompt_text += f"\n<background-context>\n{self.background_context}\n</background-context>\n"

        # Truncate transcript if needed, keeping beginning and end for context
        max_transcript_chars = 8000
        if len(transcript) > max_transcript_chars:
            half_limit = max_transcript_chars // 2
            beginning = transcript[:half_limit]
            ending = transcript[-half_limit:]
            truncated = (
                f"{beginning}\n\n"
                f"[... {len(transcript) - max_transcript_chars:,} characters omitted ...]\n\n"
                f"{ending}"
            )
        else:
            truncated = transcript

        prompt_text += f"\n<transcript>\n{truncated}\n</transcript>"

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt_text}],
            )

            yaml_text = response.content[0].text.strip()

            # Remove markdown code fence if present
            if yaml_text.startswith("```"):
                yaml_text = re.sub(r"^```(?:yaml)?\s*\n?", "", yaml_text)
                yaml_text = re.sub(r"\n?```\s*$", "", yaml_text)

            frontmatter = yaml.safe_load(yaml_text)

            # Validate required fields
            required_fields = ["title", "tags", "created"]
            for field_name in required_fields:
                if field_name not in frontmatter:
                    raise ValueError(f"Missing required field: {field_name}")

            return frontmatter

        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Failed to generate frontmatter ({e}), using fallback[/yellow]"
            )
            return {
                "title": "Workshop Transcript",
                "created": date.today().isoformat(),
                "tags": ["transcript", "workshop"],
                "description": "Transcribed workshop recording.",
            }

    @staticmethod
    def checkpoint_path(input_paths: List[Path]) -> Path:
        """Compute checkpoint file path from input file paths."""
        key = "\n".join(sorted(str(p.resolve()) for p in input_paths))
        hash8 = hashlib.sha256(key.encode()).hexdigest()[:8]
        parent = input_paths[0].resolve().parent
        return parent / f".vidscribe-checkpoint-{hash8}.json"

    def _save_checkpoint(
        self,
        path: Path,
        input_paths: List[Path],
        sections: List[TimestampSection],
        completed_batches: int,
        total_sections: int,
    ) -> None:
        """Save checkpoint atomically via write-to-temp + os.replace()."""
        data = {
            "version": 1,
            "inputs": [str(p.resolve()) for p in input_paths],
            "model": self.model,
            "batch_size": self.batch_size,
            "total_sections": total_sections,
            "completed_batches": completed_batches,
            "sections": [
                {
                    "timestamp": s.timestamp,
                    "image_embed": s.image_embed,
                    "content": s.content,
                }
                for s in sections
            ],
        }
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)

    def _load_checkpoint(
        self,
        path: Path,
        input_paths: List[Path],
    ) -> Optional[dict]:
        """Load and validate a checkpoint file."""
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        # Validate version
        if data.get("version") != 1:
            return None

        # Validate inputs match
        expected = sorted(str(p.resolve()) for p in input_paths)
        stored = sorted(data.get("inputs", []))
        if expected != stored:
            return None

        # Validate has required fields
        if "completed_batches" not in data or "sections" not in data:
            return None

        return data

    def process_all(
        self,
        document: VidcaptureDocument,
        checkpoint_path: Optional[Path] = None,
        input_paths: Optional[List[Path]] = None,
    ) -> Tuple[str, Dict]:
        """Process all sections in a vidcapture document.

        Args:
            document: Parsed vidcapture document
            checkpoint_path: Path for checkpoint file (enables resume)
            input_paths: Original input file paths (for checkpoint validation)

        Returns:
            Tuple of (filled_transcript, frontmatter_dict)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                # Validate all images exist
                valid_sections = []
                missing_count = 0
                for section in document.sections:
                    if section.image_path.exists():
                        valid_sections.append(section)
                    else:
                        missing_count += 1
                        self.console.print(
                            f"[yellow]Warning: Missing image: {section.image_path}[/yellow]"
                        )

                if missing_count > 0:
                    self.console.print(
                        f"[yellow]Skipping {missing_count} sections with missing images[/yellow]"
                    )

                if not valid_sections:
                    raise RuntimeError("No valid sections with existing images found")

                # Process in batches
                total_batches = (
                    len(valid_sections) + self.batch_size - 1
                ) // self.batch_size

                filled_sections = []
                start_batch = 0

                # Check for existing checkpoint
                if checkpoint_path and input_paths:
                    ckpt = self._load_checkpoint(checkpoint_path, input_paths)
                    if ckpt:
                        start_batch = ckpt["completed_batches"]
                        # Restore filled sections from checkpoint
                        for s_data in ckpt["sections"]:
                            for vs in valid_sections:
                                if vs.timestamp == s_data["timestamp"]:
                                    vs.content = s_data["content"]
                                    filled_sections.append(vs)
                                    break
                        self.console.print(
                            f"[bold green]Resuming from checkpoint "
                            f"({start_batch}/{total_batches} batches completed)[/bold green]"
                        )

                for batch_num in range(start_batch, total_batches):
                    start_idx = batch_num * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(valid_sections))
                    batch = valid_sections[start_idx:end_idx]

                    # Build context from previous filled sections
                    context_sections = []
                    if filled_sections and self.context_frames > 0:
                        context_sections = filled_sections[-self.context_frames:]

                    # Process batch
                    self.console.print(
                        f"\n[bold]Processing batch {batch_num + 1}/{total_batches} "
                        f"({len(batch)} sections)...[/bold]"
                    )

                    contents = self.process_markdown_batch(
                        batch,
                        context_sections,
                        temp_path,
                        progress,
                        batch_num + 1,
                        total_batches,
                    )

                    # Update sections with content
                    for i, section in enumerate(batch):
                        if i < len(contents):
                            section.content = contents[i]
                        filled_sections.append(section)

                    self.console.print(
                        f"[green]Batch {batch_num + 1}/{total_batches} complete[/green]"
                    )

                    # Save checkpoint after each batch
                    if checkpoint_path and input_paths:
                        self._save_checkpoint(
                            checkpoint_path,
                            input_paths,
                            filled_sections,
                            batch_num + 1,
                            len(valid_sections),
                        )

                # Build the full transcript
                transcript_lines = []
                for section in filled_sections:
                    transcript_lines.append(f"## {section.timestamp}")
                    transcript_lines.append(section.image_embed)
                    if section.content:
                        transcript_lines.append("")
                        transcript_lines.append(section.content)
                    transcript_lines.append("")

                transcript = "\n".join(transcript_lines)

                # Generate frontmatter
                self.console.print("\n[bold]Generating frontmatter...[/bold]")
                frontmatter = self.generate_frontmatter(transcript)

                # Clean up checkpoint on successful completion
                if checkpoint_path and checkpoint_path.exists():
                    checkpoint_path.unlink()
                    self.console.print("[dim]Checkpoint file cleaned up[/dim]")

                return transcript, frontmatter
