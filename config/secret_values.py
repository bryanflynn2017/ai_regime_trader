"""Explicit secret extraction for SDK/HTTP boundaries, never for diagnostics."""
from pydantic import SecretStr


def secret_value(value: SecretStr | str | None) -> str:
    """Support real Settings and existing plain-string test/provider overrides."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value or ""
