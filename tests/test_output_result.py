"""Human-mode output shows the underlying errors of an aggregate result."""

import logging

from vidflow.cli_common import OperationResult, output_result


def _aggregate():
    return OperationResult(
        success=False,
        message="Processed 1/2 operations",
        errors=["API request failed: gateway error 500: stack expects each tensor"],
    )


def test_errors_printed_without_logger(capsys):
    output_result(_aggregate(), json_mode=False, logger=None)
    err = capsys.readouterr().err
    assert "Error: Processed 1/2 operations" in err
    assert "  API request failed: gateway error 500" in err


def test_errors_logged_with_logger(caplog):
    logger = logging.getLogger("vidflow-test")
    with caplog.at_level(logging.ERROR, logger="vidflow-test"):
        output_result(_aggregate(), json_mode=False, logger=logger)
    messages = [r.getMessage() for r in caplog.records]
    assert "Processed 1/2 operations" in messages
    assert any("gateway error 500" in m for m in messages)


def test_error_already_in_message_is_not_repeated(capsys):
    result = OperationResult(success=False, message="Transcription failed: boom", errors=["boom"])
    output_result(result, json_mode=False, logger=None)
    err = capsys.readouterr().err
    assert err.count("boom") == 1


def test_json_mode_unchanged(capsys):
    output_result(_aggregate(), json_mode=True)
    out = capsys.readouterr().out
    assert '"errors"' in out and "gateway error 500" in out
