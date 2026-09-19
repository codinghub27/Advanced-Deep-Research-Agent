"""Environment-driven settings for official documentation retrieval.

``OFFICIAL_DOCS_ENABLED``        default true; false/0/no/off disables the source entirely
``OFFICIAL_DOCS_TIMEOUT_S``      default 15, allowed 1-60; bad values fall back to the default
``OFFICIAL_DOCS_REGISTRY_PATH``  optional JSON file merged over the built-in registry

Read at call time (not import time) so a restart is not needed to test a value, and a
malformed value can never stop the app from starting.
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
class DocsSettings:
    enabled: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    registry_path: Optional[str] = None

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "DocsSettings":
        env = os.environ if env is None else env
        enabled = env.get("OFFICIAL_DOCS_ENABLED", "true").strip().lower() not in _FALSE

        timeout_s = DEFAULT_TIMEOUT_S
        raw_timeout = env.get("OFFICIAL_DOCS_TIMEOUT_S", "").strip()
        if raw_timeout:
            try:
                value = float(raw_timeout)
            except ValueError:
                value = float("nan")
            if MIN_TIMEOUT_S <= value <= MAX_TIMEOUT_S:  # False for nan
                timeout_s = value
            else:
                logger.warning("OFFICIAL_DOCS_TIMEOUT_S must be a number between %g and %g; using %g",
                               MIN_TIMEOUT_S, MAX_TIMEOUT_S, DEFAULT_TIMEOUT_S)

        registry_path = env.get("OFFICIAL_DOCS_REGISTRY_PATH", "").strip() or None
        return cls(enabled=enabled, timeout_s=timeout_s, registry_path=registry_path)
