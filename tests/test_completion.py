"""Bash completion script stays in sync with the argparse parser."""

import argparse
import re
import shutil
import subprocess
from importlib.resources import files

import pytest

from vidflow.cli import build_parser

SCRIPT = files("vidflow.data").joinpath("completion.bash").read_text()
SUBCOMMANDS = ("youtube", "local", "transcribe", "polish")


def _subparsers() -> dict[str, argparse.ArgumentParser]:
    """Return the vidflow subcommand parsers keyed by name."""
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("vidflow parser has no subcommands")


def _script_opts(subcmd: str) -> set[str]:
    """Return the option words the script offers for *subcmd*."""
    match = re.search(rf'local {subcmd}_opts="([^"]*)"', SCRIPT)
    assert match, f"no {subcmd}_opts list in completion.bash"
    return set(match.group(1).split())


def _value_case_opts() -> set[str]:
    """Return the options handled by the script's ``case "$prev"`` block."""
    block = SCRIPT[SCRIPT.index('case "$prev" in') :]
    block = block[: block.index("\n    esac")]
    labels = re.findall(r"^\s{8}([-\w|]+)\)\s*$", block, flags=re.MULTILINE)
    return {opt for label in labels for opt in label.split("|")}


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_option_lists_match_parser(subcmd):
    sub = _subparsers()[subcmd]
    parser_opts = {s for a in sub._actions for s in a.option_strings}
    script_opts = _script_opts(subcmd)
    assert parser_opts - script_opts == set(), "missing from completion.bash"
    assert script_opts - parser_opts == set(), "stale in completion.bash"


def test_every_subcommand_is_offered():
    assert set(_subparsers()) == set(SUBCOMMANDS)
    match = re.search(r'local subcommands="([^"]*)"', SCRIPT)
    assert match and set(match.group(1).split()) == {*SUBCOMMANDS, "completion"}


def test_value_options_are_handled():
    """Options that take a value must not fall through to positional completion."""
    handled = _value_case_opts()
    for name, sub in _subparsers().items():
        for action in sub._actions:
            if action.option_strings and action.nargs != 0:
                missing = set(action.option_strings) - handled
                assert not missing, f"{name}: {sorted(missing)} not in case $prev"


def _complete(*words: str, cwd=None) -> list[str]:
    """Run the completion function for ``vidflow <words>`` and return COMPREPLY."""
    script = files("vidflow.data").joinpath("completion.bash")
    line = " ".join(["vidflow", *words])
    driver = (
        "shopt -s extglob\n"
        f'source "{script}"\n'
        'COMP_WORDS=(vidflow "$@")\n'
        "COMP_CWORD=$(( ${#COMP_WORDS[@]} - 1 ))\n"
        f"COMP_LINE={line!r}\n"
        "_vidflow_completions 2>/dev/null\n"
        'printf "%s\\n" "${COMPREPLY[@]}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", driver, "bash", *words],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=True,
    )
    return [w for w in result.stdout.splitlines() if w]


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@needs_bash
def test_completes_subcommands():
    assert set(_complete("")) == {*SUBCOMMANDS, "completion"}
    assert _complete("tr") == ["transcribe"]


@needs_bash
def test_youtube_empty_word_offers_options():
    reply = _complete("youtube", "")
    assert "--transcribe" in reply and "--keep-capture" in reply


@needs_bash
def test_subcommand_options_after_dash():
    assert "--merge" in _complete("local", "--m")
    assert "--merge" in _complete("transcribe", "--m")
    assert "--merge" not in _complete("polish", "--m")


@needs_bash
def test_positional_and_context_complete_markdown(tmp_path):
    (tmp_path / "note.md").write_text("")
    (tmp_path / "clip.mp4").write_text("")
    (tmp_path / "other.txt").write_text("")
    assert _complete("transcribe", "", cwd=tmp_path) == ["note.md"]
    assert _complete("youtube", "-c", "", cwd=tmp_path) == ["note.md"]
    assert _complete("local", "", cwd=tmp_path) == ["clip.mp4"]


@needs_bash
def test_completion_subcommand():
    assert _complete("completion", "") == ["bash"]
    assert set(_complete("completion", "bash", "--")) == {"--install", "--path"}


def test_help_documents_completion():
    from vidflow.cli import build_parser

    help_text = build_parser().format_help()
    assert "completion  " in help_text
    assert "vidflow completion bash --install" in help_text
