"""Astel blind-eval harness: corpus loader, adapters, runner, and scoring.

See ``docs/eval/CORPUS.md`` (repo root) for the frozen v1 corpus and protocol
that this package implements. That document is the source of truth; this
package's ``corpus_v1.json`` is a checked-in transcription of it, verified by
``tests/test_corpus.py`` to match on case counts, IDs, and stress tags.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
