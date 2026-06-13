# astel_eval

Blind-eval harness runner + scoring scaffold for the Astel M3 gate.

The frozen corpus and protocol live in `docs/eval/CORPUS.md` (repo root) --
that document is the source of truth. This package's `corpus_v1.json` is a
checked-in transcription verified against it by `tests/test_corpus.py`.

## Usage

```sh
uv run python -m astel_eval list
uv run python -m astel_eval run --results-dir results/
uv run python -m astel_eval score --pairwise ratings.csv
uv run python -m astel_eval gate --pairwise ratings.csv
```

All adapters (`AstelAdapter`, `Trellis2Adapter`, `MeshyAdapter`, `TripoAdapter`)
are STUBS: no network, no GPU, `available=False` on every artifact. Real
backends land post-M3.

## Gates

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src
uv run pytest
```
