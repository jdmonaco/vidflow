"""Model selection defaults and CLI argument helpers.

Single source of truth for model defaults, provider inference, and shared
argparse argument definitions. Local inference (the ampere-gateway slots,
via aikit) is the default; claude-* ids route to the Anthropic API as the
quality escape hatch.
"""

import argparse
import re

# --- Local gateway slots (the default lane) ---

DEFAULT_MODEL = "primary"  # resident long-context VLM slot
LOCAL_QUICK = "quick"  # resident secondary; titling and frontmatter

# --- Anthropic escape-hatch model IDs ---

MODEL_OPUS = "claude-opus-5"
MODEL_SONNET = "claude-sonnet-5"
MODEL_HAIKU = "claude-haiku-4-5"

# Anthropic models that reject temperature/top_p/top_k with HTTP 400: the
# 5-family (Opus/Sonnet/Fable/Mythos) and Opus 4.7/4.8. Every model at 4.6
# or earlier accepts them. Unknown claude-* ids are treated as fixed-sampling
# because omitting the param is always valid while sending it to a newer
# model is not. Local slots all accept temperature.
LAST_SAMPLING_GENERATION = (4, 6)

_VERSIONED_MODEL = re.compile(r"^claude-(?:opus|sonnet|haiku)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?$")
_LEGACY_MODEL = re.compile(r"^claude-3(?:-|$)")  # claude-3-5-sonnet-20241022 etc.


def model_accepts_temperature(model: str) -> bool:
    """True if the model accepts non-default temperature/top_p/top_k params.

    Anthropic removed sampling parameters starting with Opus 4.7 and the
    5-family; the SDK (>=1.0) no longer exposes them as keyword arguments,
    so callers must pass them via ``extra_body`` when this returns True.
    """
    if not model.startswith("claude-"):
        return True  # local gateway slots
    if _LEGACY_MODEL.match(model):
        return True
    m = _VERSIONED_MODEL.match(model)
    if m:
        generation = (int(m.group(1)), int(m.group(2) or 0))
        return generation <= LAST_SAMPLING_GENERATION
    return False  # fable-*, mythos-*, and anything unrecognised


# --- Transcription defaults ---

DEFAULT_BATCH_SIZE = 10
DEFAULT_CONTEXT_FRAMES = 3
DEFAULT_TEMPERATURE = 0.2

# Text-only polish requests carry no images, so batches can run larger
DEFAULT_POLISH_BATCH_SIZE = 20


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
