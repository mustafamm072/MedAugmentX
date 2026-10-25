"""Model assumptions and validation scope for every shipped leaf transform."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from medaugmentx.core import Transform


@lru_cache(maxsize=1)
def _cards() -> dict[str, dict[str, Any]]:
    document = json.loads(files("medaugmentx").joinpath("resources/model_cards.json").read_text())
    return document["transforms"]


def model_card(transform: str | Transform) -> dict[str, Any]:
    """Return an owned model card by built-in name or transform instance."""
    name = transform if isinstance(transform, str) else type(transform).__name__
    if name not in _cards():
        raise KeyError(f"No built-in model card for {name!r}")
    return copy.deepcopy(_cards()[name])


def list_model_cards() -> list[str]:
    """Names of all transforms with shipped model cards."""
    return sorted(_cards())
