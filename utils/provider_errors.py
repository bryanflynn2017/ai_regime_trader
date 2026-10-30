"""Allowlisted provider diagnostics: never stringify an exception or response."""
from builtins import ConnectionError as ConnectionErrorBase

from requests.exceptions import ConnectionError, Timeout


def safe_provider_error(operation: str, error: Exception) -> str:
    """operation must be a fixed application label, not provider/user content."""
    status = None
    try:
        candidate = getattr(error, "status_code", None)
        if type(candidate) is not int:
            candidate = getattr(getattr(error, "response", None), "status_code", None)
        if type(candidate) is int and 100 <= candidate <= 599:
            status = candidate
    except Exception:
        pass  # An unavailable/malformed response must not expose its contents.
    if status in (401, 403):
        detail = "AUTHENTICATION_OR_PERMISSION: check local paper credentials and permissions"
    elif status == 429:
        detail = "RATE_LIMIT: wait before retrying read requests"
    elif status == 404:
        detail = "NOT_FOUND: verify the requested resource locally"
    elif status is not None and status >= 500:
        detail = "PROVIDER_UNAVAILABLE: check service availability"
    elif status is not None and status >= 400:
        detail = "REQUEST_REJECTED: review request configuration locally"
    elif isinstance(error, (Timeout, TimeoutError)):
        detail = "TIMEOUT: check network and provider availability"
    elif isinstance(error, (ConnectionError, ConnectionErrorBase)):
        detail = "CONNECTION: check network and TLS configuration"
    else:
        detail = "PROVIDER_ERROR: check SDK/configuration and service availability"
    return f"{operation}: {detail}. Do not share raw provider responses or credentials."
