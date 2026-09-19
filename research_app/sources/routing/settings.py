"""Environment-driven settings for the source router.

``SOURCE_ROUTER_ENABLED``    default true; false/0/no/off restores the Phase 4 behaviour
                             exactly (web + official docs only; no GitHub, no Reddit, no
                             technology-problem documentation lookups)
``SOURCE_ROUTER_TIMEOUT_S``  default 15, allowed 1-60; bad values fall back to the default.
                             Applies to each GitHub / Reddit search.

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

_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class RouterSettings:
    enabled: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S

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
        return cls(enabled=enabled, timeout_s=timeout_s)
