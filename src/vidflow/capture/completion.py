"""Shell completion script generation and installation for capture commands."""

import sys
from importlib.resources import as_file, files
from pathlib import Path


def get_completion_path(command: str) -> Path:
    """Return the user-level bash completion installation path.

    Args:
        command: The command name (ytcapture or vidcapture).

    Returns:
        Path to the completion file in the user's completions directory.
    """
    return Path.home() / ".local/share/bash-completion/completions" / command


def get_bash_script_source(command: str) -> Path:
    """Return the path to the bash completion script in the package.

    Args:
        command: The command name (ytcapture or vidcapture).

    Returns:
        Resolved path to the bash script file.
    """
    resource = files("vidflow.capture.data").joinpath(f"{command}.bash")
    with as_file(resource) as path:
        return Path(path).resolve()


def get_bash_completion_script(command: str) -> str:
    """Return the bash completion script content.

    Args:
        command: The command name (ytcapture or vidcapture).

    Returns:
        The bash script content as a string.
    """
    return files("vidflow.capture.data").joinpath(f"{command}.bash").read_text()


def completion_command(command: str, args: list[str]) -> int:
    """Handle the completion subcommand.

    Usage:
        ytcapture completion bash [--install | --path]
        vidcapture completion bash [--install | --path]

    Args:
        command: The command name (ytcapture or vidcapture).
        args: Arguments after 'completion'.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    if not args or args[0] != "bash":
        print(f"Usage: {command} completion bash [--install | --path]", file=sys.stderr)
        print("Supported shells: bash", file=sys.stderr)
        return 1

    flags = args[1:] if len(args) > 1 else []

    if "--path" in flags:
        print(get_completion_path(command))
        return 0

    if "--install" in flags:
        dest = get_completion_path(command)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file/symlink
        if dest.exists() or dest.is_symlink():
            dest.unlink()

        # Create symlink to source file
        source = get_bash_script_source(command)
        dest.symlink_to(source)
        print(f"Installed: {dest} -> {source}", file=sys.stderr)
        print("Restart your shell or run: source ~/.bashrc", file=sys.stderr)
        return 0

    # Default: print to stdout
    print(get_bash_completion_script(command))
    return 0
