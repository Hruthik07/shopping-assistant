"""Add missing foreign-key indexes for frequently-joined columns.

Revision ID: 002
Revises: 001
Create Date: 2026-03-20 00:00:00.000000

Without these indexes every JOIN/filter on user_id or session_id
requires a full table scan, which degrades rapidly as row counts grow.

Note: ix_cart_items_user_id was already created in migration 001 and is
intentionally omitted here to avoid a duplicate-index error.
"""
from typing import Sequence, Union
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"], unique=False)
    op.create_index("ix_conversations_session_id", "conversations", ["session_id"], unique=False)
    op.create_index("ix_user_preferences_user_id", "user_preferences", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_user_preferences_user_id", table_name="user_preferences")
    op.drop_index("ix_conversations_session_id", table_name="conversations")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
