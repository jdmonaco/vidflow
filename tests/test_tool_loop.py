"""The Exa citation tool loop answers tool calls regardless of stop_reason.

Fable 5.1 has been observed to return tool_use blocks with stop_reason
"end_turn" on the first turn of a batch. Branching on stop_reason alone left
the calls unanswered, so the batch produced no text and every section was
reported as "No content found".
"""

import json
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


# --- Search dedup and budget exhaustion -------------------------------------

from vidflow.transcribe.prompts import (  # noqa: E402
    MAX_TOOL_CALLS_PER_BATCH,
    SEARCH_BUDGET_EXHAUSTED,
    SEARCH_FINAL_INSTRUCTION,
    SEARCH_REPEAT_PREFIX,
)


def _anthropic_processor(no_warm) -> VidscribeProcessor:
    processor = VidscribeProcessor(
        api_key="fake-key", model="claude-fable-5-1", json_output=True, exa_api_key="fake-exa"
    )
    processor._execute_exa_search = MagicMock(side_effect=lambda q: f"result for {q}")
    return processor


def test_reworded_citation_queries_are_not_searched_again(no_warm):
    processor = _anthropic_processor(no_warm)
    processor._batch_queries = []

    first = processor._search_with_dedup(
        "Malhotra Bowers 2024 background perturbations IT object recognition"
    )
    reworded = processor._search_with_dedup(
        "Malhotra Bowers 2024 deep neural network models human vision background"
    )
    quoted = processor._search_with_dedup('Malhotra Bowers 2024 "background" "IT" "object"')
    other = processor._search_with_dedup("Tanaka 2019 retinal OSR circuit prediction")

    assert first == "result for Malhotra Bowers 2024 background perturbations IT object recognition"
    assert reworded.startswith(SEARCH_REPEAT_PREFIX.split("{")[0]) and first in reworded
    assert quoted.startswith(SEARCH_REPEAT_PREFIX.split("{")[0])
    assert other == "result for Tanaka 2019 retinal OSR circuit prediction"
    assert processor._execute_exa_search.call_count == 2


def test_different_author_year_is_not_a_repeat_despite_topic_overlap(no_warm):
    processor = _anthropic_processor(no_warm)
    processor._batch_queries = []
    processor._search_with_dedup(
        "Feather et al. 2019 auditory cortex hierarchical network natural sound"
    )
    other = processor._search_with_dedup(
        "Yamins et al. 2018 auditory cortex hierarchical network natural sound"
    )
    assert other.startswith("result for Yamins")
    assert processor._execute_exa_search.call_count == 2


def test_et_al_does_not_hide_the_citation_key(no_warm):
    processor = _anthropic_processor(no_warm)
    processor._batch_queries = []
    processor._search_with_dedup("Schrimpf Kubilius 2018 neural predictivity ImageNet")
    repeat = processor._search_with_dedup("Schrimpf Kubilius et al. 2018 brain score")
    assert repeat.startswith(SEARCH_REPEAT_PREFIX.split("{")[0])
    assert processor._execute_exa_search.call_count == 1


def test_overlap_without_year_is_also_a_repeat(no_warm):
    processor = _anthropic_processor(no_warm)
    processor._batch_queries = []
    processor._search_with_dedup("hardware lottery Hooker Communications ACM")
    repeat = processor._search_with_dedup("Hooker hardware lottery Communications of the ACM")
    assert repeat.startswith(SEARCH_REPEAT_PREFIX.split("{")[0])
    assert processor._execute_exa_search.call_count == 1


def test_anthropic_lane_forces_final_turn_at_budget(no_warm, tmp_path):
    processor = _anthropic_processor(no_warm)
    sections = [_section("00:00:00")]
    final_text = "## 00:00:00\n![[img/00:00:00.jpg]]\nSlide.\n\nText."
    calls = []
    n = [0]

    def fake_request(messages, task, progress, tools=None, tool_choice=None, **kw):
        calls.append({"tool_choice": tool_choice, "messages": list(messages)})
        if tool_choice == {"type": "none"}:
            return final_text, "end_turn", _text_message(final_text)
        n[0] += 1
        tool_use = SimpleNamespace(
            type="tool_use",
            id=f"tu_{n[0]}",
            name="exa_search",
            input={"query": f"Ref{n[0]} 20{n[0]:02d} topic"},
        )
        msg = SimpleNamespace(stop_reason="tool_use", content=[tool_use])
        return "", "tool_use", msg

    processor._make_streaming_api_request = fake_request
    contents = processor.process_markdown_batch(sections, [], tmp_path, MagicMock(), 1, 1)

    assert contents == ["Slide.\n\nText."]
    assert processor._execute_exa_search.call_count == MAX_TOOL_CALLS_PER_BATCH
    assert calls[-1]["tool_choice"] == {"type": "none"}
    last_user = calls[-1]["messages"][-1]["content"]
    assert last_user[0]["content"] == SEARCH_BUDGET_EXHAUSTED
    assert last_user[-1] == {"type": "text", "text": SEARCH_FINAL_INSTRUCTION}


def test_local_lane_forces_final_turn_at_budget(no_warm, monkeypatch, tmp_path):
    processor = VidscribeProcessor(
        api_key=None, model="primary", json_output=True, exa_api_key="fake-exa"
    )
    processor._execute_exa_search = MagicMock(side_effect=lambda q: f"result for {q}")
    sections = [_section("00:00:00")]
    final_text = "## 00:00:00\n![[img/00:00:00.jpg]]\nSlide.\n\nText."
    seen = []
    n = [0]

    def fake_stream_text(client, model, messages, **kw):
        seen.append(list(messages))
        if messages[-1] == {"role": "user", "content": SEARCH_FINAL_INSTRUCTION}:
            return SimpleNamespace(text=final_text, finish_reason="stop", tool_calls=[])
        n[0] += 1
        call = {
            "id": f"tc_{n[0]}",
            "type": "function",
            "function": {
                "name": "exa_search",
                "arguments": json.dumps({"query": f"Ref{n[0]} 20{n[0]:02d} topic"}),
            },
        }
        return SimpleNamespace(text="", finish_reason="tool_calls", tool_calls=[call])

    monkeypatch.setattr("vidflow.transcribe.processor.aikit.stream_text", fake_stream_text)
    contents = processor.process_markdown_batch(sections, [], tmp_path, MagicMock(), 1, 1)

    assert contents == ["Slide.\n\nText."]
    assert processor._execute_exa_search.call_count == MAX_TOOL_CALLS_PER_BATCH
    # The over-budget call was answered with the exhausted notice before the final instruction
    tool_msgs = [m for m in seen[-1] if m.get("role") == "tool"]
    assert tool_msgs[-1]["content"] == SEARCH_BUDGET_EXHAUSTED
