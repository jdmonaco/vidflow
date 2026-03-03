"""Tests for AI-powered title generation."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from vidflow.capture.titling import (
    TitleResult,
    _clean_title,
    _validate_title,
    generate_ai_title,
    is_ai_titling_available,
)


class TestIsAiTitlingAvailable:
    """Tests for is_ai_titling_available()."""

    def test_available_with_sdk_and_key(self):
        """Available when both SDK and key are present."""
        mock_anthropic = MagicMock()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
                assert is_ai_titling_available() is True

    def test_unavailable_without_key(self):
        """Not available when ANTHROPIC_API_KEY is unset."""
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            assert is_ai_titling_available() is False

    def test_unavailable_without_sdk(self):
        """Not available when anthropic SDK is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            # Remove from sys.modules if cached
            with patch.dict(sys.modules, {"anthropic": None}):
                with patch("builtins.__import__", side_effect=mock_import):
                    assert is_ai_titling_available() is False

    def test_unavailable_with_empty_key(self):
        """Not available when ANTHROPIC_API_KEY is empty string."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            assert is_ai_titling_available() is False


class TestCleanTitle:
    """Tests for _clean_title()."""

    def test_strips_whitespace(self):
        assert _clean_title("  My Title  ") == "My Title"

    def test_strips_double_quotes(self):
        assert _clean_title('"My Title"') == "My Title"

    def test_strips_single_quotes(self):
        assert _clean_title("'My Title'") == "My Title"

    def test_strips_hash_marks(self):
        assert _clean_title("## My Title") == "My Title"

    def test_quotes_then_hash(self):
        result = _clean_title('"## My Title"')
        assert result == "My Title"

    def test_preserves_normal_title(self):
        assert _clean_title("Ilya Sutskever - Scaling Research") == "Ilya Sutskever - Scaling Research"

    def test_empty_string(self):
        assert _clean_title("") == ""


def _make_mock_anthropic(response_text):
    """Create a mock anthropic module with a preconfigured response."""
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=response_text)]
    mock_client.messages.create.return_value = mock_response
    return mock_module, mock_client


class TestValidateTitle:
    """Tests for _validate_title()."""

    def test_valid_title(self):
        assert _validate_title("Ilya Sutskever - Moving from Scaling to Research") is True

    def test_too_few_words(self):
        assert _validate_title("Title") is False

    def test_too_many_words(self):
        assert _validate_title("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen") is False

    def test_too_short(self):
        assert _validate_title("A - B C D") is False

    def test_too_long(self):
        assert _validate_title("A" * 151) is False

    def test_exactly_two_words(self):
        assert _validate_title("Reasonable Title") is True

    def test_twelve_words(self):
        assert _validate_title("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve") is True


class TestGenerateAiTitle:
    """Tests for generate_ai_title()."""

    def test_successful_generation(self):
        """Successful API call produces a valid AI title."""
        mock_module, mock_client = _make_mock_anthropic(
            "Ilya Sutskever - Scaling Neural Networks"
        )

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            result = generate_ai_title(
                title="NVIDIA GTC 2025: Ilya Sutskever on Why Scaling Neural Networks Changed Everything",
                channel="NVIDIA",
                description="In this talk at GTC 2025...",
            )

        assert result.used_ai is True
        assert result.ai_title == "Ilya Sutskever - Scaling Neural Networks"
        assert "NVIDIA GTC" in result.original_title

    def test_validation_failure_falls_back(self):
        """Invalid AI output falls back to original title."""
        mock_module, mock_client = _make_mock_anthropic("X")  # Too short

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            result = generate_ai_title(
                title="Original Title Here",
                channel="Channel",
                description="Description",
            )

        assert result.used_ai is False
        assert result.ai_title == "Original Title Here"

    def test_api_timeout_falls_back(self):
        """API timeout falls back to original title."""
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = TimeoutError("Request timed out")

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            result = generate_ai_title(
                title="Original Title",
                channel="Channel",
                description="Desc",
            )

        assert result.used_ai is False
        assert result.ai_title == "Original Title"

    def test_auth_error_falls_back(self):
        """Authentication error falls back to original title."""
        mock_module = MagicMock()
        mock_client = MagicMock()
        mock_module.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("Invalid API key")

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            result = generate_ai_title(
                title="Original Title",
                channel="Channel",
                description="Desc",
            )

        assert result.used_ai is False
        assert result.ai_title == "Original Title"

    def test_import_error_falls_back(self):
        """Missing anthropic SDK falls back to original title."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return real_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"anthropic": None}):
            with patch("builtins.__import__", side_effect=mock_import):
                result = generate_ai_title(
                    title="Original Title",
                    channel="Channel",
                    description="Desc",
                )

        assert result.used_ai is False
        assert result.ai_title == "Original Title"

    def test_quoted_output_cleaned(self):
        """Quoted LLM output is cleaned before validation."""
        mock_module, mock_client = _make_mock_anthropic(
            '"John Smith - Deep Learning Basics"'
        )

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            result = generate_ai_title(
                title="Some Long Original Title",
                channel="Channel",
                description="Desc",
            )

        assert result.used_ai is True
        assert result.ai_title == "John Smith - Deep Learning Basics"

    def test_description_truncated(self):
        """Description longer than 500 chars is truncated in the API call."""
        mock_module, mock_client = _make_mock_anthropic(
            "Jane Doe - Research Overview"
        )

        long_desc = "x" * 1000

        with patch.dict(sys.modules, {"anthropic": mock_module}):
            result = generate_ai_title(
                title="Title",
                channel="Channel",
                description=long_desc,
            )

        # Verify the API was called with truncated description
        call_args = mock_client.messages.create.call_args
        user_msg = call_args[1]["messages"][0]["content"]
        assert "x" * 501 not in user_msg
        assert result.used_ai is True


class TestTitleResult:
    """Tests for TitleResult dataclass."""

    def test_dataclass_fields(self):
        result = TitleResult(ai_title="AI Title", original_title="Original", used_ai=True)
        assert result.ai_title == "AI Title"
        assert result.original_title == "Original"
        assert result.used_ai is True


class TestMarkdownTruncation:
    """Tests for the 10-word truncation in markdown filename generation."""

    def test_ten_word_truncation(self):
        from vidflow.capture.utils import truncate_title_words

        title = "One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve"
        result = truncate_title_words(title, 10)
        assert result == "One Two Three Four Five Six Seven Eight Nine Ten"

    def test_short_title_unchanged(self):
        from vidflow.capture.utils import truncate_title_words

        title = "Short Title"
        result = truncate_title_words(title, 10)
        assert result == "Short Title"


class TestOriginalTitleFrontmatter:
    """Tests for original_title in frontmatter when AI title is used."""

    def test_original_title_in_frontmatter(self):
        from vidflow.capture.markdown import generate_frontmatter
        from vidflow.capture.video import VideoMetadata

        metadata = VideoMetadata(
            video_id="abc123",
            title="AI Generated Title Here",
            channel="Test Channel",
            upload_date="20240101",
            description="Test description",
            duration=120.0,
            _original_title="Original Very Long YouTube Title With SEO Keywords",
        )
        fm = generate_frontmatter(metadata)
        assert "original_title:" in fm
        assert "Original Very Long YouTube Title" in fm

    def test_no_original_title_when_same(self):
        from vidflow.capture.markdown import generate_frontmatter
        from vidflow.capture.video import VideoMetadata

        metadata = VideoMetadata(
            video_id="abc123",
            title="Same Title",
            channel="Test Channel",
            upload_date="20240101",
            description="Test description",
            duration=120.0,
            _original_title="Same Title",
        )
        fm = generate_frontmatter(metadata)
        assert "original_title:" not in fm

    def test_no_original_title_when_empty(self):
        from vidflow.capture.markdown import generate_frontmatter
        from vidflow.capture.video import VideoMetadata

        metadata = VideoMetadata(
            video_id="abc123",
            title="Some Title",
            channel="Test Channel",
            upload_date="20240101",
            description="Test description",
            duration=120.0,
        )
        fm = generate_frontmatter(metadata)
        assert "original_title:" not in fm
