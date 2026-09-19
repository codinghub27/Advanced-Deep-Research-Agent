import logging
import os


def configure_logging() -> None:
    """
    Make application logger.info() output visible (uvicorn only configures its
    own loggers). Safe to call more than once: basicConfig does nothing if the
    root logger already has handlers. Level comes from LOG_LEVEL (default INFO).
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
