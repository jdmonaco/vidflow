"""Model selection defaults and CLI argument helpers.

Single source of truth for Claude model IDs, transcription defaults,
and shared argparse argument definitions.
"""

import argparse

# --- Model IDs ---

MODEL_OPUS_47 = "claude-opus-4-7"
MODEL_OPUS = "claude-opus-4-6"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-4-5"

# Opus 4.6 remains the default: 4.7 rejects non-default sampling params, and
# transcription benefits from the low-temperature control we use.
TRANSCRIBE_MODELS = [MODEL_OPUS, MODEL_OPUS_47, MODEL_SONNET, MODEL_HAIKU]

# Models that reject non-default temperature/top_p/top_k with HTTP 400.
# The API call must omit these params when using a model in this set.
FIXED_SAMPLING_MODELS = frozenset({MODEL_OPUS_47})


def model_accepts_temperature(model: str) -> bool:
    """True if the model accepts non-default temperature/top_p/top_k params."""
    return model not in FIXED_SAMPLING_MODELS


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
