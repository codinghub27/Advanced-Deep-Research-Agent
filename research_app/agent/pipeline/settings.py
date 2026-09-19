"""Environment-driven settings for the research pipeline (Phase 6).

``CONCURRENCY_LIMIT``     default 6, allowed 1-20: source searches in flight at once, across
                          the whole process (every Tavily call takes one slot)
``GAP_SEARCH_ENABLED``    default true: false skips the (single) gap-filling search round
``MAX_TARGETED_QUERIES``  default 2, allowed 0-4: queries in the gap round, and in a retry
``MAX_CRITIC_RETRIES``    default 1, allowed 0-1: 0 disables the retry. The upper bound is the
                          design (one retry), not a tunable
``RELEVANCE_THRESHOLD``   default 0.3, allowed 0-1: share of a sub-question's content words a
                          document must contain to count as evidence for it

Read at call time; malformed values fall back to their defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from research_app.conversation.settings import read_bool, read_float, read_int


@dataclass(frozen=True)
class PipelineSettings:
    concurrency_limit: int = 6
    gap_search_enabled: bool = True
    max_targeted_queries: int = 2
    max_critic_retries: int = 1
    relevance_threshold: float = 0.3

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "PipelineSettings":
        env = os.environ if env is None else env
        return cls(
            concurrency_limit=read_int(env, "CONCURRENCY_LIMIT", 6, 1, 20),
            gap_search_enabled=read_bool(env, "GAP_SEARCH_ENABLED", True),
            max_targeted_queries=read_int(env, "MAX_TARGETED_QUERIES", 2, 0, 4),
            max_critic_retries=read_int(env, "MAX_CRITIC_RETRIES", 1, 0, 1),
            relevance_threshold=read_float(env, "RELEVANCE_THRESHOLD", 0.3, 0.0, 1.0),
        )
