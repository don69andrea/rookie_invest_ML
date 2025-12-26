from __future__ import annotations

from collections.abc import Iterable


def require_columns(columns: Iterable[str], required: Iterable[str], context: str) -> None:
    required_set = set(required)
    missing = required_set.difference(columns)
    if missing:
        raise ValueError(f"Missing required columns for {context}: {sorted(missing)}")
