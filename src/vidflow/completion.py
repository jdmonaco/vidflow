"""Shell completion script generation and installation."""

import sys
from importlib.resources import as_file, files
from pathlib import Path


def get_completion_path() -> Path:
    """Return the user-level bash completion installation path."""
    return Path.home() / ".local/share/bash-completion/completions/vidflow"


def get_bash_script_source() -> Path:
    """Return the path to the bash completion script in the package."""
    resource = files("vidflow.data").joinpath("completion.bash")
    with as_file(resource) as path:
        return Path(path).resolve()


def get_bash_completion_script() -> str:
    """Return the bash completion script content."""
    return files("vidflow.data").joinpath("completion.bash").read_text()


def completion_command(args: list[str]) -> int:
    """Handle the completion subcommand.

    Usage:
        vidflow completion bash [--install | --path]

    Args:
        args: Arguments after 'completion'.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    if not args or args[0] != "bash":
        print("Usage: vidflow completion bash [--install | --path]", file=sys.stderr)
        print("Supported shells: bash", file=sys.stderr)
        return 1

    flags = args[1:] if len(args) > 1 else []

    if "--path" in flags:
        print(get_completion_path())
        return 0

    if "--install" in flags:
        dest = get_completion_path()
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() or dest.is_symlink():
            dest.unlink()

        source = get_bash_script_source()
        dest.symlink_to(source)
        print(f"Installed: {dest} -> {source}", file=sys.stderr)
        print("Restart your shell or run: source ~/.bashrc", file=sys.stderr)
        return 0

    # Default: print to stdout
    print(get_bash_completion_script())
    return 0
