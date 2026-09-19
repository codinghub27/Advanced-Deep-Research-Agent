"""Environment-driven settings for the source router.

``SOURCE_ROUTER_ENABLED``    default true; false/0/no/off restores the Phase 4 behaviour
                             exactly (web + official docs only; no GitHub, no Reddit, no
                             technology-problem documentation lookups)
``SOURCE_ROUTER_TIMEOUT_S``  default 15, allowed 1-60; bad values fall back to the default.
                             Applies to each GitHub / Reddit search.
``MAX_SOURCES_PER_TASK``     default 3, allowed 1-4 (Phase 6): the most sources one sub-question is
                             sent to. Official documentation and web are never trimmed by it.

Read at call time (not import time), like the official-docs settings, so a malformed value
can never stop the app from starting.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 15.0
MIN_TIMEOUT_S = 1.0
MAX_TIMEOUT_S = 60.0
DEFAULT_MAX_SOURCES_PER_TASK = 3
MIN_MAX_SOURCES_PER_TASK = 1
MAX_MAX_SOURCES_PER_TASK = 4  # there are only four source types

_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RouterSettings:
    enabled: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_sources_per_task: int = DEFAULT_MAX_SOURCES_PER_TASK

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RouterSettings":
        env = os.environ if env is None else env
        enabled = env.get("SOURCE_ROUTER_ENABLED", "true").strip().lower() not in _FALSE

        timeout_s = DEFAULT_TIMEOUT_S
        raw = env.get("SOURCE_ROUTER_TIMEOUT_S", "").strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                value = float("nan")
            if MIN_TIMEOUT_S <= value <= MAX_TIMEOUT_S:  # False for nan
                timeout_s = value
            else:
                logger.warning("SOURCE_ROUTER_TIMEOUT_S must be a number between %g and %g; using %g",
                               MIN_TIMEOUT_S, MAX_TIMEOUT_S, DEFAULT_TIMEOUT_S)

        cap = DEFAULT_MAX_SOURCES_PER_TASK
        raw = env.get("MAX_SOURCES_PER_TASK", "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError:
                value = 0
            if MIN_MAX_SOURCES_PER_TASK <= value <= MAX_MAX_SOURCES_PER_TASK:
                cap = value
            else:
                logger.warning("MAX_SOURCES_PER_TASK must be an integer between %d and %d; using %d",
                               MIN_MAX_SOURCES_PER_TASK, MAX_MAX_SOURCES_PER_TASK, DEFAULT_MAX_SOURCES_PER_TASK)
        return cls(enabled=enabled, timeout_s=timeout_s, max_sources_per_task=cap)
