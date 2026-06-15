"""billing columns (mode, refine_of, credits)

Adds the preview/refine billing fields to ``generations`` (M3, CLAUDE.md §7).

Revision ID: a1b2c3d4e5f6
Revises: c2f332907e2c
Create Date: 2026-06-15 12:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: str | Sequence[str] | None = 'c2f332907e2c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'generations',
        sa.Column(
            'mode',
            sa.String(length=16),
            nullable=False,
            server_default='refine',
        ),
    )
    op.add_column(
        'generations',
        sa.Column('refine_of', sa.String(length=36), nullable=True),
    )
    op.add_column(
        'generations',
        sa.Column('credits', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('generations', 'credits')
    op.drop_column('generations', 'refine_of')
    op.drop_column('generations', 'mode')
