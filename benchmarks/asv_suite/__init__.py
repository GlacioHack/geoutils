"""Benchmark setup for ASV."""

from __future__ import annotations

import os


def asv_pr_check_enabled() -> bool:
    """Whether ASV should use the lightweight inputs (to reduce ``asv check`` time to ~10 min for CI quick PR test)."""

    return os.environ.get("GEOUTILS_ASV_PR_CHECK") == "1"
