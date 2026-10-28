"""
pytest configuration and shared fixtures.

The pyproject.toml sets pythonpath = ["."] which adds the project root to
sys.path, so imports like `from config.settings import settings` work when
running pytest from within the ai_regime_trader/ directory.
"""
from __future__ import annotations

import pytest
