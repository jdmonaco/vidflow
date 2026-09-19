"""Tests for tiered frontmatter model selection in VidscribeProcessor."""

from unittest.mock import MagicMock

import pytest

from vidflow.transcribe.processor import VidscribeProcessor

VALID_YAML = (
    "title: Test Talk on Neural Data\n"
    "created: 2026-08-29\n"
    "tags:\n  - test\n"
    "description: A test transcript.\n"
)


@pytest.fixture
def no_warm(monkeypatch):
    monkeypatch.setattr("vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None)


def _local_response(text: str) -> MagicMock:
    response = MagicMock()
    response.choices[0].message.content = text
    return response


def _anthropic_response(text: str) -> MagicMock:
    response = MagicMock()
    response.content[0].text = text
    return response


class TestLocalLane:
    """Local lane: frontmatter runs on the resident session model."""

    def test_uses_session_model(self, no_warm):
        processor = VidscribeProcessor(
            api_key=None, model="primary", json_output=True, text_only=True
        )
        processor.local_client = MagicMock()
        processor.local_client.chat.completions.create.return_value = _local_response(VALID_YAML)

        frontmatter = processor.generate_frontmatter("some transcript")

        call_kwargs = processor.local_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "primary"
        assert frontmatter["title"] == "Test Talk on Neural Data"

    def test_failure_falls_back_to_static(self, no_warm):
        processor = VidscribeProcessor(
            api_key=None, model="primary", json_output=True, text_only=True
        )
        processor.local_client = MagicMock()
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )

        frontmatter = processor.generate_frontmatter("some transcript")

        assert frontmatter["title"] == "Workshop Transcript"


class TestYamlRepair:
    """Model YAML with bare colons or flow openers is repaired, not discarded."""

    def _processor(self, no_warm_unused) -> VidscribeProcessor:
        processor = VidscribeProcessor(
            api_key=None, model="primary", json_output=True, text_only=True
        )
        processor.local_client = MagicMock()
        return processor

    def test_repairs_title_with_colon(self, no_warm):
        # The exact response that produced the "Workshop Transcript" fallback
        bad = (
            "title: Physical Intelligence: General-Purpose Robots for the Real World\n"
            "created: 2026-09-19\n"
            "tags:\n  - robotics\n"
            "description: Chelsea Finn's talk: data, scale, and RL for robot policies.\n"
        )
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.return_value = _local_response(bad)

        frontmatter = processor.generate_frontmatter("some transcript")

        assert frontmatter["title"] == (
            "Physical Intelligence: General-Purpose Robots for the Real World"
        )
        assert frontmatter["description"].startswith("Chelsea Finn's talk: data")
        assert frontmatter["tags"] == ["robotics"]

    def test_repair_leaves_valid_yaml_alone(self):
        assert VidscribeProcessor.repair_yaml_scalars(VALID_YAML) == VALID_YAML

    def test_repair_quotes_flow_openers_and_escapes(self):
        text = 'title: [Draft] A "quoted" word: here\nkey: plain value\n'
        repaired = VidscribeProcessor.repair_yaml_scalars(text)
        import yaml

        data = yaml.safe_load(repaired)
        assert data["title"] == '[Draft] A "quoted" word: here'
        assert data["key"] == "plain value"

    def test_fallback_uses_capture_title(self, no_warm):
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.side_effect = RuntimeError("down")

        frontmatter = processor.generate_frontmatter("t", fallback_title="Capture Title")

        assert frontmatter["title"] == "Capture Title"


class TestAnthropicLane:
    """Anthropic lane: quick slot first, session model second, static last."""

    def _processor(self, no_warm_unused, model="claude-opus-5") -> VidscribeProcessor:
        processor = VidscribeProcessor(
            api_key="fake-key", model=model, json_output=True, text_only=True
        )
        processor.local_client = MagicMock()
        processor.client = MagicMock()
        return processor

    def test_quick_slot_preferred(self, no_warm):
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.return_value = _local_response(VALID_YAML)

        frontmatter = processor.generate_frontmatter("some transcript")

        call_kwargs = processor.local_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "quick"
        assert frontmatter["title"] == "Test Talk on Neural Data"
        processor.client.messages.create.assert_not_called()

    def test_quick_failure_falls_back_to_session_model(self, no_warm):
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )
        processor.client.messages.create.return_value = _anthropic_response(VALID_YAML)

        frontmatter = processor.generate_frontmatter("some transcript")

        call_kwargs = processor.client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-5"
        # Fixed-sampling model: temperature must be omitted
        assert "temperature" not in call_kwargs
        assert frontmatter["title"] == "Test Talk on Neural Data"

    def test_fallback_passes_temperature_when_supported(self, no_warm):
        processor = self._processor(no_warm, model="claude-haiku-4-5")
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )
        processor.client.messages.create.return_value = _anthropic_response(VALID_YAML)

        processor.generate_frontmatter("some transcript")

        call_kwargs = processor.client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.1

    def test_both_lanes_failing_uses_static_fallback(self, no_warm):
        processor = self._processor(no_warm)
        processor.local_client.chat.completions.create.side_effect = RuntimeError(
            "admission_refused"
        )
        processor.client.messages.create.side_effect = RuntimeError("api down")

        frontmatter = processor.generate_frontmatter("some transcript")

        assert frontmatter["title"] == "Workshop Transcript"
