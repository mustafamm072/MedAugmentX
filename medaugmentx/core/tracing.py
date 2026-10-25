"""Per-context execution tracing; no global pipeline state or I/O."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable

ACTIVE_TRACE: ContextVar[Callable[[Any, str], None] | None] = ContextVar(
    "medaugmentx_trace", default=None
)
