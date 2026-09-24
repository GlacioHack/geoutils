"""Expose repeatable benchmarks and select the lightweight pull-request profile."""

from __future__ import annotations

import os


def asv_pr_check_enabled() -> bool:
    """Whether ASV should use the lightweight pull-request inputs."""

    return os.environ.get("GEOUTILS_ASV_PR_CHECK") == "1"
