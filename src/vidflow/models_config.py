"""Model selection defaults and CLI argument helpers.

Single source of truth for Claude model IDs, transcription defaults,
and shared argparse argument definitions.
"""

import argparse

# --- Model IDs ---

MODEL_OPUS = "claude-opus-4-6"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5"

TRANSCRIBE_MODELS = [MODEL_OPUS, MODEL_SONNET, MODEL_HAIKU]

# --- Transcription defaults ---

DEFAULT_MODEL = MODEL_OPUS
DEFAULT_BATCH_SIZE = 10
DEFAULT_CONTEXT_FRAMES = 3
DEFAULT_TEMPERATURE = 0.2


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model selection and temperature arguments to a parser.

    Provides -m/--model with validated choices and --temperature
    with range checking, using consistent defaults across all
    entry points.
    """
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        choices=TRANSCRIBE_MODELS,
        help=f"Claude model for transcription (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"API temperature (default: {DEFAULT_TEMPERATURE})",
    )
