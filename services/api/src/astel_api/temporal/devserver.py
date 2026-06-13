"""Helper to start/stop ``temporal server start-dev`` as a managed subprocess.

This module is intentionally NOT imported anywhere at module-import time (not
from ``engine.py``, not from ``main.py``) so that the absence of the
``temporal`` CLI binary never breaks imports or the default offline test
suite. It is for ``astel up``-style local dev orchestration only.

Per the spike (docs/research/10-task-engine-spike.md), the binary is a
standalone ``temporal``/``temporal.exe`` with zero install — we locate it on
``PATH`` (or an explicit override) and shell out to ``server start-dev``.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_FRONTEND_ADDRESS = "localhost:7233"


def _resolve_binary(binary: str | None = None) -> str:
    """Find the ``temporal`` CLI, raising if it cannot be located."""
    candidate = binary or shutil.which("temporal") or shutil.which("temporal.exe")
    if candidate is None:
        raise FileNotFoundError(
            "temporal CLI binary not found on PATH; download it from "
            "https://github.com/temporalio/cli/releases (see "
            "docs/research/10-task-engine-spike.md) and either add it to "
            "PATH or pass binary=... explicitly."
        )
    return candidate


def start_dev_server(
    *,
    binary: str | None = None,
    db_filename: str | Path | None = None,
    frontend_address: str = DEFAULT_FRONTEND_ADDRESS,
    namespace: str = "default",
    extra_args: list[str] | None = None,
) -> subprocess.Popen[bytes]:
    """Launch ``temporal server start-dev`` and return the running process.

    The caller owns the returned process: terminate it (or use
    :func:`dev_server` as a context manager) to stop the server. ``db_filename``
    enables sqlite persistence across restarts, matching the spike's
    ``--db-filename`` flag.
    """
    exe = _resolve_binary(binary)
    args = [exe, "server", "start-dev", "--ip", "127.0.0.1"]

    host, _, port = frontend_address.rpartition(":")
    if host:
        args += ["--ip", host]
    if port:
        args += ["--port", port]
    if namespace and namespace != "default":
        args += ["--namespace", namespace]
    if db_filename is not None:
        args += ["--db-filename", str(db_filename)]
    if extra_args:
        args += extra_args

    return subprocess.Popen(args)  # noqa: S603 - intentional managed subprocess


@contextmanager
def dev_server(
    *,
    binary: str | None = None,
    db_filename: str | Path | None = None,
    frontend_address: str = DEFAULT_FRONTEND_ADDRESS,
    namespace: str = "default",
    extra_args: list[str] | None = None,
) -> Iterator[str]:
    """Context manager: start the dev server, yield its frontend address, stop it."""
    proc = start_dev_server(
        binary=binary,
        db_filename=db_filename,
        frontend_address=frontend_address,
        namespace=namespace,
        extra_args=extra_args,
    )
    try:
        yield frontend_address
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
