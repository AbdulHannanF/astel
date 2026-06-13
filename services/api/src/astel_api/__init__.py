"""Astel API gateway package.

M1 skeleton: FastAPI app exposing health, generation submission, and an SSE
event stream that simulates the L0->L3 layer pipeline. The task layer sits
behind a :class:`~astel_api.engine.TaskEngine` interface so the in-process stub
can be swapped for Temporal next session without touching the routes.
"""

__version__ = "0.1.0"
