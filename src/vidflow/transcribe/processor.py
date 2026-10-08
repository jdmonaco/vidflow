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
from urllib.parse import urlparse

import aikit
import anthropic
import yaml
from anthropic import Anthropic
from rich.console import Console
from rich.markup import escape
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
    POLISH_PROMPT,
    SEARCH_BUDGET_EXHAUSTED,
    SEARCH_FINAL_INSTRUCTION,
    SEARCH_MAX_REPEATS,
    SEARCH_NO_CANONICAL_URL,
    SEARCH_NO_RESULTS,
    SEARCH_REPEAT_PREFIX,
    SEARCH_REPEAT_THRESHOLD,
    TEMPLATE_FILL_PROMPT,
)

# Optional Exa citation search support
try:
    from exa_py import Exa as ExaClient

    EXA_AVAILABLE = True
except ImportError:
    EXA_AVAILABLE = False


# A timestamp section heading as the batch template and response use it
_TIMESTAMP_HEADING_RE = re.compile(r"^##\s+\d{2}:\d{2}:\d{2}\s*$", re.MULTILINE)


class VidscribeProcessor:
    """Main processor for video frame transcription."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        temperature: float = 0.2,
        batch_size: int = 10,
        context_frames: int = 3,
        max_dimension: int = 1568,
        background_context: str = "",
        json_output: bool = False,
        exa_api_key: Optional[str] = None,
        provider: Optional[str] = None,
        text_only: bool = False,
    ):
        """Initialize the processor.

        The provider is inferred from the model name (claude-* routes to
        Anthropic, anything else to the local gateway) unless overridden.
        api_key is only required on the Anthropic lane.

        text_only enables polish mode: no frame images are sent and the
        POLISH_PROMPT replaces the vision template-fill prompt, so only the
        caption text in each section is cleaned up. Citation search is a
        vision feature (slide references) and is disabled in this mode.
        """
        from vidflow.models_config import DEFAULT_TEMPERATURE, model_accepts_temperature

        self.provider = aikit.provider_for(model, provider)
        self.model = model
        self.temperature = temperature
        self.batch_size = batch_size
        self.context_frames = context_frames
        self.max_dimension = max_dimension
        self.background_context = background_context
        self.json_output = json_output
        self.text_only = text_only
        self.console = Console(quiet=json_output)
        self.magick_cmd = None if text_only else find_magick_command()

        # Local client serves the local lane and frontmatter generation
        # (see _request_frontmatter for the model-selection tiers)
        self.local_client = aikit.local_client()
        if self.provider == "anthropic":
            self.client = Anthropic(api_key=api_key)
            self.supports_temperature = model_accepts_temperature(model)
        else:
            self.client = None
            self.supports_temperature = True  # local slots accept temperature
            aikit.warm(model)  # overlap any JIT load with image prep

        if not self.supports_temperature and temperature != DEFAULT_TEMPERATURE:
            self.console.print(
                f"[yellow]Warning:[/] model {model} rejects non-default sampling "
                f"parameters; --temperature {temperature} will be ignored."
            )

        # Exa citation search (vision mode only). _batch_queries holds
        # (tokens, query, result) for searches made in the current batch.
        self._batch_queries: list[tuple[set, str, str]] = []
        self._batch_repeats = 0
        self.exa_enabled = False
        self.exa_client = None
        if exa_api_key and not text_only:
            if EXA_AVAILABLE:
                self.exa_client = ExaClient(api_key=exa_api_key)
                self.exa_enabled = True
            else:
                self.console.print(
                    "[yellow]Warning: EXA_API_KEY is set but exa-py is not installed. "
                    "Reinstall vidflow (exa-py is a core dependency)[/yellow]"
                )

    def estimate_tokens(self, sections: List[TimestampSection]) -> int:
        """Estimate total input tokens for a list of sections."""
        prompt_tokens = len(TEMPLATE_FILL_PROMPT) // 4
        if self.exa_enabled:
            prompt_tokens += len(CITATION_SEARCH_PROMPT) // 4
        context_tokens = len(self.background_context) // 4 if self.background_context else 0
        # Conservative middle estimate; polish mode sends no images
        image_tokens = 0 if self.text_only else len(sections) * 1200
        existing_text_tokens = sum(len(s.existing_text) // 4 for s in sections if s.existing_text)
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
        tool_choice: dict | None = None,
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
                    messages=messages,
                )
                if self.supports_temperature:
                    # anthropic>=1.0 dropped the temperature kwarg; extra_body
                    # merges into the request JSON on every SDK version
                    stream_kwargs["extra_body"] = {"temperature": self.temperature}
                if tools:
                    stream_kwargs["tools"] = tools
                if tool_choice:
                    stream_kwargs["tool_choice"] = tool_choice

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
                verb = "Polishing" if self.text_only else "Transcribing"
                progress.update(
                    progress_task,
                    description=f"{verb} batch with {self.model}",
                )

            except Exception as e:
                raise RuntimeError(f"API request failed: {str(e)}")

        raise RuntimeError("All retry attempts failed")

    def _run_local_batch(self, messages: list, progress_task, progress) -> str:
        """Local-gateway lane: stream a batch with tool-use loop.

        aikit.stream_text retries transient gateway errors (409/503) with
        Retry-After budgets and resumes truncation via user-turn
        continuations internally. Streaming tool-call deltas are verified
        working through the shim (qwen3_coder parser on primary).
        """
        tools = None
        if self.exa_enabled:
            tools = [
                aikit.tool_def(
                    EXA_SEARCH_TOOL["name"],
                    EXA_SEARCH_TOOL["description"],
                    EXA_SEARCH_TOOL["input_schema"],
                )
            ]

        def on_delta(text: str, _seen=[0]) -> None:
            _seen[0] += len(text)
            progress.update(progress_task, completed=min(95, _seen[0] / 20))

        response_text = ""
        tool_call_count = 0
        forced_final = False
        while True:
            try:
                result = aikit.stream_text(
                    self.local_client,
                    model=self.model,
                    messages=messages,
                    max_tokens=16000,
                    temperature=self.temperature,
                    # The forced final turn offers no tools at all: aikit has
                    # no tool_choice passthrough, and with the tool still
                    # offered the model intermittently answered the "write
                    # the transcript" instruction with no template
                    tools=None if forced_final else tools,
                    on_delta=on_delta,
                    max_continuations=3,
                )
            except (aikit.GatewayTransient, aikit.GatewayPermanent) as e:
                raise RuntimeError(f"API request failed: {e}")

            response_text += result.text

            pending = bool(
                result.finish_reason == "tool_calls" and self.exa_enabled and result.tool_calls
            )
            if pending and not forced_final:
                exhausted = self._search_budget_exhausted(tool_call_count)
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.text or None,
                        "tool_calls": result.tool_calls,
                    }
                )
                for tc in result.tool_calls:
                    if exhausted:
                        tool_content = SEARCH_BUDGET_EXHAUSTED
                    else:
                        tool_call_count += 1
                        try:
                            args = json.loads(tc["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        tool_content = self._search_with_dedup(args.get("query", ""))
                    messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": tool_content}
                    )
                if exhausted:
                    # Pending calls answered; demand the transcript in one
                    # final turn, sent without tool definitions
                    self.console.print(self._budget_warning())
                    messages.append({"role": "user", "content": SEARCH_FINAL_INSTRUCTION})
                    forced_final = True
                progress.update(
                    progress_task,
                    description=f"Processing citations ({tool_call_count} searches)",
                    completed=0,
                )
                continue

            if pending and forced_final:
                self.console.print(
                    "[yellow]Warning: Model kept requesting searches after the budget "
                    "was exhausted; batch may be empty[/yellow]"
                )
            if result.finish_reason == "length":
                self.console.print(
                    "[yellow]Warning: Response still truncated after "
                    "continuation attempts[/yellow]"
                )
            progress.update(progress_task, completed=100)
            return response_text

    @staticmethod
    def _query_tokens(query: str) -> set:
        """Lowercase alphanumeric tokens of three or more characters."""
        return {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}

    @staticmethod
    def _citation_key(query: str) -> str | None:
        """Leading 'Author [Author] Year' tokens of a citation-style query.

        Returns None when no year appears within the first four tokens
        ("et al" is ignored).
        """
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in ("et", "al")]
        for i, tok in enumerate(tokens[:4]):
            if re.fullmatch(r"(19|20)\d\d", tok) and i >= 1:
                return " ".join(tokens[: i + 1])
        return None

    def _search_budget_exhausted(self, tool_call_count: int) -> bool:
        """True once the batch has spent its searches or kept repeating them."""
        return (
            tool_call_count >= MAX_TOOL_CALLS_PER_BATCH or self._batch_repeats >= SEARCH_MAX_REPEATS
        )

    def _budget_warning(self) -> str:
        if self._batch_repeats >= SEARCH_MAX_REPEATS:
            reason = f"{self._batch_repeats} repeated searches"
        else:
            reason = f"tool call limit ({MAX_TOOL_CALLS_PER_BATCH})"
        return (
            f"[yellow]Warning: Hit {reason} for this batch; requesting the "
            "transcript without further searches[/yellow]"
        )

    def _search_with_dedup(self, query: str) -> str:
        """Search unless this batch already searched for the same reference.

        A repeat is a query with the same leading author-year key as an
        earlier one, or, when either query lacks such a key, with token
        overlap at or above SEARCH_REPEAT_THRESHOLD. Two queries with
        different author-year keys are different references however much
        their topic words overlap. Repeats return the earlier result with
        an instruction not to reword and retry, without calling Exa.
        """
        tokens = self._query_tokens(query)
        key = self._citation_key(query)
        for prev_tokens, prev_query, prev_result in self._batch_queries:
            prev_key = self._citation_key(prev_query)
            if key is not None and prev_key is not None:
                is_repeat = key == prev_key
            else:
                union = tokens | prev_tokens
                overlap = len(tokens & prev_tokens) / len(union) if union else 0.0
                is_repeat = overlap >= SEARCH_REPEAT_THRESHOLD
            if is_repeat:
                self._batch_repeats += 1
                self.console.print(f"[dim]  Repeat query (not searched): {query[:80]}[/dim]")
                return SEARCH_REPEAT_PREFIX.format(previous=prev_query) + prev_result
        self.console.print(
            f"[dim]  Searching: {query[:80]}...[/dim]"
            if len(query) > 80
            else f"[dim]  Searching: {query}[/dim]"
        )
        result = self._execute_exa_search(query)
        self._batch_queries.append((tokens, query, result))
        return result

    # Hosts that are the search index's own landing pages, not the paper
    _INDEX_HOSTS = ("exa.ai",)
    _DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>)\]]+)")

    @classmethod
    def _is_canonical_url(cls, url: str | None) -> bool:
        """True for a publisher, DOI, preprint, or author-hosted link."""
        if not url:
            return False
        host = urlparse(url).netloc.lower()
        return not any(host == h or host.endswith("." + h) for h in cls._INDEX_HOSTS)

    def _execute_exa_search(self, query: str) -> str:
        """Execute an Exa academic paper search and return formatted result.

        Prefers the first result with a canonical URL; an index landing page
        (exa.ai/library/...) is reported without a URL. A DOI found in any
        result's excerpt is reported as a doi.org link.
        """
        try:
            results = self.exa_client.search_and_contents(
                query,
                type="auto",
                category="research paper",
                num_results=3,
                text={"max_characters": 500},
            )

            if not results.results:
                return SEARCH_NO_RESULTS.format(query=query)

            r = next(
                (x for x in results.results if self._is_canonical_url(x.url)),
                results.results[0],
            )
            doi = next(
                (
                    m.group(1)
                    for x in results.results
                    for m in [self._DOI_RE.search(x.text or "")]
                    if m
                ),
                None,
            )
            parts = []
            if r.title:
                parts.append(f"Title: {r.title}")
            if self._is_canonical_url(r.url):
                parts.append(f"URL: {r.url}")
            else:
                parts.append(SEARCH_NO_CANONICAL_URL)
            if doi:
                parts.append(f"DOI: https://doi.org/{doi.rstrip('.')}")
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
        if self.text_only:
            return POLISH_PROMPT
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
                    f"<existing-transcript>\n" f"{sec.existing_text}\n" f"</existing-transcript>\n"
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

        A response with no timestamp sections at all is a failed batch, not
        an empty one (observed after the citation search budget ran out:
        the forced final turn intermittently returned no template). It is
        retried once with citation search disabled.

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
        args = (sections, previous_sections, temp_dir, progress, batch_num, total_batches)
        response_text = self._request_batch(*args)
        if sections and not _TIMESTAMP_HEADING_RE.search(response_text):
            excerpt = escape(repr(response_text.strip()[:200]))
            self.console.print(
                f"[yellow]Warning: Batch {batch_num}/{total_batches} response has no "
                f"timestamp sections ({len(response_text)} chars: {excerpt}); "
                "retrying without citation search[/yellow]",
                highlight=False,
            )
            exa_enabled, self.exa_enabled = self.exa_enabled, False
            try:
                response_text = self._request_batch(*args)
            finally:
                self.exa_enabled = exa_enabled
        return self._parse_batch_response(response_text, sections)

    def _request_batch(
        self,
        sections: list[TimestampSection],
        previous_sections: list[TimestampSection],
        temp_dir: Path,
        progress,
        batch_num: int,
        total_batches: int,
    ) -> str:
        """Build the batch request, run it (with the citation tool loop), return raw text."""
        self._batch_queries = []
        self._batch_repeats = 0
        content = []

        # Images first (per Claude Vision API guidance: images before text);
        # polish mode is text-only and sends no frame images at all
        if not self.text_only:
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
                content.append({"type": "text", "text": f"\n[Frame: {section.timestamp}]\n"})

                # Add image (provider-specific block shape)
                if self.provider == "anthropic":
                    image_block = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": self.get_media_type(prepared_path),
                            "data": base64_image,
                        },
                    }
                else:
                    image_block = aikit.image_part(base64_image, self.get_media_type(prepared_path))
                content.append(image_block)

            # Close the frame-images tag
            content.append({"type": "text", "text": "\n</frame-images>\n\n"})

        # Add background context if available. Anthropic gets an explicit
        # cache_control breakpoint; the local SGLang backend radix-caches
        # prefixes server-side on its own, so no field is needed there.
        if self.background_context:
            context_block = {
                "type": "text",
                "text": f"<background-context>\n{self.background_context}\n</background-context>\n\n",
            }
            if self.provider == "anthropic":
                context_block["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
            content.append(context_block)

        # Add previous transcription context if available
        if previous_sections:
            context_text = "<previous-transcription-context>\n"
            context_text += (
                "The following are the most recent transcribed sections. Use for continuity:\n\n"
            )
            for sec in previous_sections:
                context_text += f"## {sec.timestamp}\n{sec.image_embed}\n{sec.content}\n\n"
            context_text += "</previous-transcription-context>\n\n"
            content.append({"type": "text", "text": context_text})

        # Add the template-fill prompt and template
        content.append({"type": "text", "text": self._get_batch_prompt()})
        content.append({"type": "text", "text": self._build_batch_template(sections)})

        # Make API request
        verb = "Polishing" if self.text_only else "Transcribing"
        api_task = progress.add_task(
            f"{verb} batch {batch_num}/{total_batches} with {self.model}",
            total=100,
        )

        messages = [{"role": "user", "content": content}]

        # Local lane: aikit handles streaming, retry, tool loop, continuation
        if self.provider == "local":
            response_text = self._run_local_batch(messages, api_task, progress)
            progress.remove_task(api_task)
            return response_text

        # Pass tools if Exa citation search is enabled
        tools = [EXA_SEARCH_TOOL] if self.exa_enabled else None

        response_text, stop_reason, final_message = self._make_streaming_api_request(
            messages, api_task, progress, tools=tools
        )

        # Handle tool use loop (Exa citation searches). Branch on the
        # presence of tool_use blocks rather than stop_reason: Fable 5.1 has
        # been observed to return tool_use blocks with stop_reason "end_turn",
        # and an unanswered tool call yields an empty batch.
        tool_call_count = 0
        while self.exa_enabled:
            tool_use_blocks = [block for block in final_message.content if block.type == "tool_use"]
            if not tool_use_blocks:
                break

            # Echo the assistant turn back verbatim: thinking blocks must
            # precede tool_use blocks when continuing a tool-use turn
            messages.append({"role": "assistant", "content": final_message.content})

            # Past the budget, answer the pending calls with a refusal and
            # force one final text turn with tool_choice none
            exhausted = self._search_budget_exhausted(tool_call_count)
            tool_results = []
            for block in tool_use_blocks:
                if exhausted:
                    result = SEARCH_BUDGET_EXHAUSTED
                else:
                    tool_call_count += 1
                    result = self._search_with_dedup(block.input.get("query", ""))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

            user_content: list = tool_results
            tool_choice = None
            if exhausted:
                self.console.print(self._budget_warning())
                user_content = tool_results + [{"type": "text", "text": SEARCH_FINAL_INSTRUCTION}]
                tool_choice = {"type": "none"}
            messages.append({"role": "user", "content": user_content})

            progress.update(
                api_task,
                description=f"Processing citations ({tool_call_count} searches)",
                completed=0,
            )

            continued_text, stop_reason, final_message = self._make_streaming_api_request(
                messages, api_task, progress, tools=tools, tool_choice=tool_choice
            )
            response_text += continued_text
            if exhausted:
                break

        # Handle continuation if truncated (multi-turn, no prefill)
        max_continuations = 3
        continuation_count = 0

        while (
            stop_reason in ["max_tokens", "pause_turn"] and continuation_count < max_continuations
        ):
            continuation_count += 1
            self.console.print(
                f"[yellow]Response truncated ({stop_reason}), continuing... "
                f"(attempt {continuation_count}/{max_continuations})[/yellow]"
            )

            # Echo the API-returned content blocks back verbatim
            messages.append({"role": "assistant", "content": final_message.content})

            # Append a user-role continuation prompt
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response was truncated. Continue the transcript "
                        "exactly where you left off. Do not repeat any content already "
                        "provided."
                    ),
                }
            )

            progress.update(
                api_task,
                description=f"Continuing transcription ({continuation_count})",
                completed=0,
            )

            continued_text, stop_reason, final_message = self._make_streaming_api_request(
                messages, api_task, progress
            )
            response_text += continued_text

        if stop_reason in ["max_tokens", "pause_turn"]:
            self.console.print(
                f"[yellow]Warning: Response still truncated after "
                f"{max_continuations} continuation attempts[/yellow]"
            )

        # Remove the task to prevent accumulation in the progress display
        progress.remove_task(api_task)
        return response_text

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

    def _request_frontmatter(self, prompt_text: str) -> str:
        """Request the frontmatter YAML from the best available model.

        On the local lane the session model is used directly: it is the
        resident backend that just served the batches, so admission cannot
        be refused — the quick slot, by contrast, may not fit alongside it
        (gateway 409 admission_refused when VRAM is tight). On the
        Anthropic lane the free local quick slot is tried first, falling
        back to the session's Anthropic model.

        Reasoning tokens count against max_tokens on the local models, so
        leave generous headroom above the ~15-line YAML output.
        """
        from vidflow.models_config import LOCAL_QUICK

        messages = [{"role": "user", "content": prompt_text}]

        if self.provider == "local":
            response = self.local_client.chat.completions.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.1,
                messages=messages,
            )
            return response.choices[0].message.content or ""

        try:
            response = self.local_client.chat.completions.create(
                model=LOCAL_QUICK,
                max_tokens=1500,
                temperature=0.1,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Local quick slot unavailable for frontmatter "
                f"({e}), using {self.model}[/yellow]"
            )
            kwargs = dict(
                model=self.model,
                max_tokens=1500,
                messages=messages,
            )
            if self.supports_temperature:
                kwargs["extra_body"] = {"temperature": 0.1}
            response = self.client.messages.create(**kwargs)
            return response.content[0].text

    @staticmethod
    def repair_yaml_scalars(yaml_text: str) -> str:
        """Quote top-level scalar values a model emitted as bare YAML.

        Models routinely write ``title: Foo: Bar`` (a second ``: `` opens a
        nested mapping) or ``description: [Draft] ...`` (a ``[`` opens flow
        syntax that then fails to close). A top-level ``key: value`` whose
        value is not already a quoted or block scalar, and either contains a
        mapping separator or does not parse on its own, is wrapped in double
        quotes with YAML escaping. Valid inline lists, nested blocks, and
        list items pass through untouched.
        """
        fixed = []
        for line in yaml_text.splitlines():
            match = re.match(r"^([A-Za-z_][\w-]*):[ \t]+(.+?)[ \t]*$", line)
            if match:
                key, value = match.groups()
                if VidscribeProcessor._yaml_value_needs_quoting(value):
                    value = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
                    line = f"{key}: {value}"
            fixed.append(line)
        trailing = "\n" if yaml_text.endswith("\n") else ""
        return "\n".join(fixed) + trailing

    @staticmethod
    def _yaml_value_needs_quoting(value: str) -> bool:
        if value[0] in "|>":
            return False  # block scalar indicator
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            return False  # already quoted
        if ": " in value or value.endswith(":") or " #" in value:
            return True
        try:
            parsed = yaml.safe_load(f"k: {value}")
        except yaml.YAMLError:
            return True
        return not isinstance(parsed, dict) or isinstance(parsed.get("k"), dict)

    def generate_frontmatter(self, transcript: str, fallback_title: Optional[str] = None) -> Dict:
        """Generate YAML frontmatter based on transcript content.

        ``fallback_title`` (normally the capture note's own title) names the
        output when the model's YAML cannot be salvaged, so a bad response
        never yields a generic filename the user then has to rename.
        """
        # Build the prompt
        prompt_text = FRONTMATTER_PROMPT
        prompt_text += f"\n\nToday's date: {date.today().isoformat()}\n"

        if self.background_context:
            prompt_text += (
                f"\n<background-context>\n{self.background_context}\n</background-context>\n"
            )

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
            yaml_text = self._request_frontmatter(prompt_text).strip()

            # Remove markdown code fence if present
            if yaml_text.startswith("```"):
                yaml_text = re.sub(r"^```(?:yaml)?\s*\n?", "", yaml_text)
                yaml_text = re.sub(r"\n?```\s*$", "", yaml_text)

            try:
                frontmatter = yaml.safe_load(yaml_text)
            except yaml.YAMLError:
                frontmatter = yaml.safe_load(self.repair_yaml_scalars(yaml_text))
            if not isinstance(frontmatter, dict):
                raise ValueError("response is not a YAML mapping")

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
                "title": fallback_title or "Workshop Transcript",
                "created": date.today().isoformat(),
                "tags": ["transcript"],
                "description": "Transcribed recording.",
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

    @staticmethod
    def _part_heading(section: TimestampSection, total_parts: int) -> str:
        """Rich markup naming the input file a batch belongs to."""
        source = section.source_path
        name = source.name if source else (section.part_title or f"Part {section.part_index + 1}")
        prefix = f"File {section.part_index + 1}/{total_parts}" if total_parts > 1 else "File"
        return f"[bold cyan]{prefix}:[/bold cyan] [bold]{escape(name)}[/bold]"

    def process_all(
        self,
        document: VidcaptureDocument,
        checkpoint_path: Optional[Path] = None,
        input_paths: Optional[List[Path]] = None,
        with_frontmatter: bool = True,
    ) -> Tuple[str, Dict]:
        """Process all sections in a vidcapture document.

        Args:
            document: Parsed vidcapture document
            checkpoint_path: Path for checkpoint file (enables resume)
            input_paths: Original input file paths (for checkpoint validation)
            with_frontmatter: Generate frontmatter from the result; False
                skips the model call and returns an empty dict (used by
                in-place polish, which keeps the original frontmatter)

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
                # Validate all images exist (polish mode never reads them)
                if self.text_only:
                    valid_sections = list(document.sections)
                else:
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

                # Process in batches that never straddle part boundaries:
                # timestamps restart per merged part, so keeping each batch
                # within one part keeps `## HH:MM:SS` headings unique in the
                # template and response, and keeps continuity context from
                # leaking across unrelated recordings.
                batches: List[List[TimestampSection]] = []
                prev_part: Optional[int] = None
                for section in valid_sections:
                    if (
                        not batches
                        or section.part_index != prev_part
                        or len(batches[-1]) >= self.batch_size
                    ):
                        batches.append([])
                    batches[-1].append(section)
                    prev_part = section.part_index
                total_batches = len(batches)

                filled_sections = []
                start_batch = 0

                # Check for existing checkpoint
                if checkpoint_path and input_paths:
                    ckpt = self._load_checkpoint(checkpoint_path, input_paths)
                    if ckpt:
                        # Restore filled sections in processing order; merged
                        # parts repeat timestamps, so positional restore with a
                        # sanity check replaces timestamp lookup
                        restored = ckpt["sections"]
                        if all(
                            vs.timestamp == s_data["timestamp"]
                            for vs, s_data in zip(valid_sections, restored)
                        ):
                            start_batch = ckpt["completed_batches"]
                            for vs, s_data in zip(valid_sections, restored):
                                vs.content = s_data["content"]
                                filled_sections.append(vs)
                            self.console.print(
                                f"[bold green]Resuming from checkpoint "
                                f"({start_batch}/{total_batches} batches completed)[/bold green]"
                            )
                        else:
                            self.console.print(
                                "[yellow]Checkpoint does not match inputs; starting fresh[/yellow]"
                            )

                # Announce each input file before its first batch; batches
                # never straddle parts, so a part change marks a new file
                total_parts = len({s.part_index for s in valid_sections})
                announced_part: int | None = None

                for batch_num in range(start_batch, total_batches):
                    batch = batches[batch_num]

                    if batch[0].part_index != announced_part:
                        announced_part = batch[0].part_index
                        self.console.print(
                            "\n" + self._part_heading(batch[0], total_parts), highlight=False
                        )

                    # Build context from previous filled sections of the SAME
                    # part; a new part starts with no continuity context
                    context_sections = []
                    if filled_sections and self.context_frames > 0:
                        context_sections = [
                            s
                            for s in filled_sections[-self.context_frames :]
                            if s.part_index == batch[0].part_index
                        ]

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

                # Build the full transcript. A merged run emits an H1 per
                # original file, with H2 timestamps restarting under each.
                multi_part = len({s.part_index for s in filled_sections}) > 1
                transcript_lines = []
                heading_part: Optional[int] = None
                for section in filled_sections:
                    if multi_part and section.part_index != heading_part:
                        part_title = section.part_title or f"Part {section.part_index + 1}"
                        transcript_lines.append(f"# {part_title}")
                        transcript_lines.append("")
                        heading_part = section.part_index
                    transcript_lines.append(f"## {section.timestamp}")
                    transcript_lines.append(section.image_embed)
                    if section.content:
                        transcript_lines.append("")
                        transcript_lines.append(section.content)
                    transcript_lines.append("")

                transcript = "\n".join(transcript_lines)

                # Generate frontmatter (skipped for in-place polish)
                frontmatter: Dict = {}
                if with_frontmatter:
                    self.console.print("\n[bold]Generating frontmatter...[/bold]")
                    frontmatter = self.generate_frontmatter(transcript, document.title)

                # Clean up checkpoint on successful completion
                if checkpoint_path and checkpoint_path.exists():
                    checkpoint_path.unlink()
                    self.console.print("[dim]Checkpoint file cleaned up[/dim]")

                return transcript, frontmatter
