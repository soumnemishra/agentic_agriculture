"""
ACA Logging & Tracing
=====================

Provides structured logging with optional trace-ID propagation for
the Agricultural Cognitive Architecture. Every subsystem obtains its
logger through ``get_logger()``, ensuring consistent formatting and
centralised level control.

Trace IDs allow correlating log entries that belong to the same
cognitive reasoning cycle across agents, memory, and orchestration.
"""

from __future__ import annotations

import logging
import threading
import uuid
from contextvars import ContextVar
from typing import Optional

from aca.config import LoggingConfig

# ---------------------------------------------------------------------------
# Trace-ID context variable (async-safe via contextvars)
# ---------------------------------------------------------------------------
_trace_id: ContextVar[Optional[str]] = ContextVar("aca_trace_id", default=None)

_setup_done = False
_setup_lock = threading.Lock()


def setup_logging(config: LoggingConfig) -> None:
    """
    Initialise the ACA logging subsystem.

    Should be called once at application startup. Subsequent calls are
    no-ops (guarded by a lock).

    Args:
        config: Logging configuration specifying level, format, and
                optional file output.
    """
    global _setup_done
    with _setup_lock:
        if _setup_done:
            return

        root = logging.getLogger("aca")
        root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

        formatter = logging.Formatter(config.format)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        if config.log_file:
            file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

        _setup_done = True


def get_logger(name: str) -> logging.Logger:
    """
    Obtain a child logger under the ``aca`` namespace.

    Args:
        name: Dot-separated logger name (e.g. ``orchestration.message_bus``).

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(f"aca.{name}")


# ---------------------------------------------------------------------------
# Trace-ID helpers
# ---------------------------------------------------------------------------

def new_trace_id() -> str:
    """Generate a fresh trace ID and store it in the current context."""
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> Optional[str]:
    """Return the current trace ID, or ``None`` if not set."""
    return _trace_id.get()


def set_trace_id(tid: str) -> None:
    """Manually set the trace ID for the current context."""
    _trace_id.set(tid)
