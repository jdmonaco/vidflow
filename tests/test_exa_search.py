"""Exa search results never hand the model an index landing page as the URL."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vidflow.transcribe.processor import VidscribeProcessor
from vidflow.transcribe.prompts import SEARCH_NO_CANONICAL_URL


@pytest.fixture
def processor(monkeypatch):
    monkeypatch.setattr("vidflow.transcribe.processor.aikit.warm", lambda *a, **k: None)
    p = VidscribeProcessor(
        api_key="fake-key", model="claude-fable-5-1", json_output=True, exa_api_key="fake-exa"
    )
    p.exa_client = MagicMock()
    return p


def _result(url, title="Emergent World Models", text="", author="Li, K.", date="2023-06-01"):
    return SimpleNamespace(url=url, title=title, text=text, author=author, published_date=date)


def _search(processor, *results):
    processor.exa_client.search_and_contents.return_value = SimpleNamespace(results=list(results))
    return processor._execute_exa_search("Li 2023 Othello-GPT")


def test_prefers_first_canonical_url_over_index_page(processor):
    out = _search(
        processor,
        _result("https://exa.ai/library/publication/8zgj8pcdlf6"),
        _result("https://arxiv.org/abs/2210.13382"),
    )
    assert "URL: https://arxiv.org/abs/2210.13382" in out
    assert "exa.ai" not in out


def test_index_only_results_report_no_url(processor):
    out = _search(processor, _result("https://exa.ai/library/publication/abc"))
    assert "URL:" not in out
    assert SEARCH_NO_CANONICAL_URL in out
    assert "Title: Emergent World Models" in out


def test_doi_in_excerpt_is_reported(processor):
    out = _search(
        processor,
        _result(
            "https://www.exa.ai/library/x", text="See doi:10.1038/s41586-024-07000-1. Abstract..."
        ),
    )
    assert "DOI: https://doi.org/10.1038/s41586-024-07000-1" in out
    assert "URL:" not in out


@pytest.mark.parametrize(
    "url, canonical",
    [
        ("https://doi.org/10.1/x", True),
        ("https://journals.plos.org/plosone/article?id=1", True),
        ("https://exa.ai/library/publication/1", False),
        ("https://www.exa.ai/library/publication/1", False),
        ("https://notexa.ai/paper", True),
        (None, False),
    ],
)
def test_is_canonical_url(url, canonical):
    assert VidscribeProcessor._is_canonical_url(url) is canonical
