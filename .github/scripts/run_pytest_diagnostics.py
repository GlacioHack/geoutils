"""Print pytest exit status and Python shutdown stages for CI diagnostics."""

from __future__ import annotations

import atexit
import faulthandler
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path


def _print_stage(message: str) -> None:
    """Flush a diagnostic message to the original stderr stream."""

    print(f"\n[pytest diagnostics] {message}", file=sys.__stderr__, flush=True)


def main() -> int:
    """Run pytest and report its status before Python shuts down."""

    # Match module execution so repository-only packages remain importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _print_stage(f"Python: {sys.version}")
    _print_stage(f"Executable: {sys.executable}; PID: {os.getpid()}")

    # This callback runs after callbacks registered while importing pytest and its dependencies
    atexit.register(_print_stage, "Shutdown reached the callback registered before importing pytest")
    import pytest

    class ExitDiagnostics:
        """Print test counts and exit status around pytest cleanup hooks."""

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> Iterator[None]:
            """Print test failures and status before and after session cleanup."""

            _print_stage(
                f"Session finish started: exitstatus={int(exitstatus)}, "
                f"testsfailed={session.testsfailed}, testscollected={session.testscollected}"
            )
            try:
                yield
            finally:
                _print_stage(
                    f"Session finish completed: exitstatus={int(session.exitstatus)}, testsfailed={session.testsfailed}"
                )

        @pytest.hookimpl(hookwrapper=True, tryfirst=True)
        def pytest_unconfigure(self, config: pytest.Config) -> Iterator[None]:
            """Print when plugin cleanup starts and finishes."""

            _print_stage("Plugin cleanup started")
            try:
                yield
            finally:
                _print_stage("Plugin cleanup completed")

    # Preserve pytest arguments and its return code while exposing the final cleanup stages
    result = pytest.main(sys.argv[1:], plugins=[ExitDiagnostics()])
    _print_stage(f"pytest.main() returned: {int(result)}")
    threads = [(thread.name, thread.daemon, thread.ident) for thread in threading.enumerate()]
    _print_stage(f"Threads before shutdown (name, daemon, ident): {threads}")

    # Leave fault reporting active during shutdown and dump stacks if it takes more than ten seconds
    faulthandler.enable()
    faulthandler.dump_traceback_later(10, repeat=True)
    atexit.register(_print_stage, "Shutdown started running callbacks registered with atexit")
    return int(result)


if __name__ == "__main__":
    sys.exit(main())
