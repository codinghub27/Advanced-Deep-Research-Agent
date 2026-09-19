API_LIMIT_MESSAGE = "API limit reached. Please try again later."


def is_groq_rate_limit(exc: BaseException) -> bool:
    try:
        from groq import RateLimitError

        if isinstance(exc, RateLimitError):
            return True
    except ImportError:
        pass

    current: BaseException | None = exc
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if type(current).__name__ == "RateLimitError":
            return True

        status_code = getattr(current, "status_code", None)
        if status_code == 429:
            return True

        response = getattr(current, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True

        message = str(current).lower()
        if any(
            token in message
            for token in ("rate limit", "429", "too many requests", "quota")
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


def api_limit_response() -> dict:
    return {"api_limit_reached": True, "final_answer": API_LIMIT_MESSAGE}
