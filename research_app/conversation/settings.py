"""Environment-driven settings for conversation persistence (Phase 6).

``CONVERSATION_HISTORY_LIMIT``  default 10, allowed 1-50: turns loaded as context
``ANSWER_SUMMARY_LENGTH``       default 300, allowed 50-2000: characters of an earlier answer
                                kept in the context (never the full answer)
``SESSION_TITLE_MAX_LENGTH``    default 80, allowed 10-255: auto-generated session title

Read at call time; a malformed value falls back to its default and logs a warning, so it can
never stop the app from starting. Pure: no database or framework imports.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


def read_int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = low - 1
    if low <= value <= high:
        return value
    logger.warning("%s must be an integer between %d and %d; using %d", name, low, high, default)
    return default


def read_float(env: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if low <= value <= high:  # False for nan
        return value
    logger.warning("%s must be a number between %g and %g; using %g", name, low, high, default)
    return default


def read_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ConversationSettings:
    history_limit: int = 10
    answer_summary_length: int = 300
    title_max_length: int = 80

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ConversationSettings":
        env = os.environ if env is None else env
        return cls(
            history_limit=read_int(env, "CONVERSATION_HISTORY_LIMIT", 10, 1, 50),
            answer_summary_length=read_int(env, "ANSWER_SUMMARY_LENGTH", 300, 50, 2000),
            title_max_length=read_int(env, "SESSION_TITLE_MAX_LENGTH", 80, 10, 255),
        )
