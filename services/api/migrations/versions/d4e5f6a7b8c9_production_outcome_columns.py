"""production outcome + conditioning columns

Adds the fields that couple the SSE engine to the real production outcome
(audit §2.6/§2.7) and the structured ``conditioning`` field (audit
recommendation #2):

- ``produced``: whether produce_artifacts_dispatch actually wrote artifacts.
- ``splats``: the real splat count from the producer's result dict.
- ``production_error``: str(exc) on production failure, else null.
- ``conditioning``: what the L3 geometry was conditioned on for this task
  ("prompt" | "image" | "video" | "none").

Revision ID: d4e5f6a7b8c9
Revises: a1b2c3d4e5f6
Create Date: 2026-06-15 13:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: str | Sequence[str] | None = 'a1b2c3d4e5f6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'generations',
        sa.Column(
            'produced',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        'generations',
        sa.Column('splats', sa.Integer(), nullable=True),
    )
    op.add_column(
        'generations',
        sa.Column('production_error', sa.Text(), nullable=True),
    )
    op.add_column(
        'generations',
        sa.Column('conditioning', sa.String(length=16), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('generations', 'conditioning')
    op.drop_column('generations', 'production_error')
    op.drop_column('generations', 'splats')
    op.drop_column('generations', 'produced')
