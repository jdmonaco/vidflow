"""The Exa citation tool loop answers tool calls regardless of stop_reason.

Fable 5.1 has been observed to return tool_use blocks with stop_reason
"end_turn" on the first turn of a batch. Branching on stop_reason alone left
the calls unanswered, so the batch produced no text and every section was
reported as "No content found".
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vidflow.transcribe.models import TimestampSection
from vidflow.transcribe.processor import VidscribeProcessor


@pytest.fixture
def no_warm(monkeypatch):
    monkeypatch.setattr("vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None)


def _section(ts: str) -> TimestampSection:
    return TimestampSection(
        timestamp=ts,
        image_embed=f"![[img/{ts}.jpg]]",
        image_path=MagicMock(exists=lambda: False),
        existing_text="raw caption text",
    )


def _tool_message(stop_reason: str):
    tool_use = SimpleNamespace(
        type="tool_use", id="tu_1", name="search_citations", input={"query": "q"}
    )
    thinking = SimpleNamespace(type="thinking", thinking="")
    return SimpleNamespace(stop_reason=stop_reason, content=[thinking, tool_use])


def _text_message(text: str):
    return SimpleNamespace(
        stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)]
    )


@pytest.mark.parametrize("first_stop_reason", ["tool_use", "end_turn"])
def test_tool_calls_answered_regardless_of_stop_reason(no_warm, first_stop_reason, tmp_path):
    processor = VidscribeProcessor(
        api_key="fake-key", model="claude-fable-5-1", json_output=True, exa_api_key="fake-exa"
    )
    assert processor.exa_enabled
    processor._execute_exa_search = MagicMock(return_value="search results")

    sections = [_section("00:00:00")]
    first = _tool_message(first_stop_reason)
    final_text = "## 00:00:00\n![[img/00:00:00.jpg]]\nSlide.\n\n**Speaker**: Cleaned text."
    responses = iter(
        [("", first.stop_reason, first), (final_text, "end_turn", _text_message(final_text))]
    )
    sent = []

    def fake_request(messages, task, progress, tools=None, **kw):
        sent.append([m["role"] for m in messages])
        return next(responses)

    processor._make_streaming_api_request = fake_request

    contents = processor.process_markdown_batch(sections, [], tmp_path, MagicMock(), 1, 1)

    processor._execute_exa_search.assert_called_once_with("q")
    assert contents == ["Slide.\n\n**Speaker**: Cleaned text."]
    # Second request carries the echoed assistant turn (thinking + tool_use) and the tool result
    assert sent[1] == ["user", "assistant", "user"]
