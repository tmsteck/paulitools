"""Numba compatibility helpers for paulitools.

Editable installs that map the package directly to ``src/`` can fail during
import on newer Python/Numba stacks when ``@njit(cache=True)`` cannot locate a
stable package cache path.  Keep caching opt-in through an environment variable
so the package remains importable in modern development environments.
"""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


NUMBA_CACHE = _env_flag("PAULITOOLS_NUMBA_CACHE", default=False)
