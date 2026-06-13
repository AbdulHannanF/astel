"""Temporal-backed task engine subpackage.

Importing this package (and ``shared``) must never require a running Temporal
server or even a reachable ``temporal`` binary — only ``devserver`` and
``worker`` touch a live connection, and only when actually run.
"""

from __future__ import annotations
