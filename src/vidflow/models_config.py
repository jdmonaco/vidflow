"""Model selection defaults and CLI argument helpers.

Single source of truth for model defaults, provider inference, and shared
argparse argument definitions. Local inference (the ampere-gateway slots,
via aikit) is the default; claude-* ids route to the Anthropic API as the
quality escape hatch.
"""

import argparse

# --- Local gateway slots (the default lane) ---

DEFAULT_MODEL = "primary"  # resident long-context VLM slot
LOCAL_QUICK = "quick"  # resident secondary; titling and frontmatter

# --- Anthropic escape-hatch model IDs ---

MODEL_OPUS = "claude-opus-5"
MODEL_SONNET = "claude-sonnet-5"
MODEL_HAIKU = "claude-haiku-4-5"

# Anthropic models that reject non-default temperature/top_p/top_k with
# HTTP 400: the 5-family and Opus 4.7/4.8. The API call must omit these
# params when using a model in this set. Local slots all accept temperature.
FIXED_SAMPLING_MODELS = frozenset({
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
})


def model_accepts_temperature(model: str) -> bool:
    """True if the model accepts non-default temperature/top_p/top_k params."""
    return model not in FIXED_SAMPLING_MODELS


# --- Transcription defaults ---

DEFAULT_BATCH_SIZE = 10
DEFAULT_CONTEXT_FRAMES = 3
DEFAULT_TEMPERATURE = 0.2


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model selection and temperature arguments to a parser.

    -m/--model is deliberately open (no choices=): gateway slot ids must
    survive server-side evolution, and a typo'd slot yields a clean 404
    from the gateway. --provider forces a lane when inference from the
    model name is not wanted.
    """
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=(
            f"Model: a gateway slot id (primary, quick, ...) for local "
            f"inference, or a claude-* id (e.g. {MODEL_OPUS}) routed to "
            f"Anthropic (default: {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--provider",
        choices=("local", "anthropic"),
        default=None,
        help="Force a provider instead of inferring it from the model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
