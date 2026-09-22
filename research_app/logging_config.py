import logging
import os

# Modules that log one INFO line per graph node / routing decision / search. They are the
# "echo of every node" in the server terminal, so they are quiet by default and come back with
# LOG_NODE_TRACE=true. Warnings and errors from them are always shown. Per-request lines
# (research_app.main, research_app.research_service) stay at LOG_LEVEL.
NODE_TRACE_LOGGERS = ("research_app.agent", "research_app.sources")


class _HealthCheckFilter(logging.Filter):
    """Drops uvicorn's access lines for ``GET /health``: the UI polls it every 15 seconds."""

    def filter(self, record: logging.LogRecord) -> bool:
        return '"GET /health ' not in record.getMessage()


def configure_logging() -> None:
    """
    Make application logger.info() output visible (uvicorn only configures its
    own loggers). Safe to call more than once: basicConfig does nothing if the
    root logger already has handlers. Level comes from LOG_LEVEL (default INFO).

    Per-node and per-search INFO lines and the /health access lines are hidden unless
    LOG_NODE_TRACE is true (default false).
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # HTTP client libraries log full request URLs at INFO; keep them quiet.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # LOG_LEVEL=DEBUG asks for detail, so it turns the trace on too.
    trace = level <= logging.DEBUG or os.getenv("LOG_NODE_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}
    for name in NODE_TRACE_LOGGERS:
        # Never make a logger chattier than LOG_LEVEL asks for.
        logging.getLogger(name).setLevel(logging.NOTSET if trace else max(level, logging.WARNING))

    access = logging.getLogger("uvicorn.access")
    access.filters = [f for f in access.filters if not isinstance(f, _HealthCheckFilter)]
    if not trace:
        access.addFilter(_HealthCheckFilter())
