"""Common CLI infrastructure for vidflow.

Provides shared utilities for consistent CLI behavior including
logging setup, exit codes, and structured output for agentic workflows.
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Optional


class ExitCode(IntEnum):
    """Standard exit codes."""

    SUCCESS = 0
    ERROR = 1
    USAGE_ERROR = 2


@dataclass
class OperationResult:
    """Structured result for JSON output in agentic workflows.

    Attributes:
        success: Whether the operation completed successfully.
        message: Human-readable summary of the operation.
        data: Optional structured data (counts, file paths, etc.).
        errors: Optional list of error messages.
    """

    success: bool
    message: str
    data: Optional[dict[str, Any]] = None
    errors: Optional[list[str]] = None

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to JSON string."""
        return json.dumps(asdict(self), indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    name: str = "vidflow",
) -> logging.Logger:
    """Configure logging based on verbosity flags.

    Args:
        verbose: If True, set level to DEBUG.
        quiet: If True, set level to WARNING.
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def add_common_args(parser) -> None:
    """Add standard flags to any argparse parser.

    Adds -v/--verbose, -q/--quiet, and --json flags.
    """
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress non-error output")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )


def output_result(
    result: OperationResult,
    json_mode: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Output result in appropriate format.

    In JSON mode, prints the result as JSON to stdout.
    Otherwise, logs success/failure message to stderr.
    """
    if json_mode:
        print(result.to_json())
    else:
        if logger:
            if result.success:
                logger.info(result.message)
            else:
                logger.error(result.message)
        else:
            if result.success:
                print(result.message, file=sys.stderr)
            else:
                print(f"Error: {result.message}", file=sys.stderr)
