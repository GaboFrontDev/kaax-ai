"""create agent_knowledge table

Revision ID: 20260228_000003
Revises: 20260227_000002
Create Date: 2026-02-28 00:00:03
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260228_000003"
down_revision = "20260227_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_knowledge",
        sa.Column("knowledge_id", sa.Text(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("agent_id", sa.Text(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("TRUE")),
        sa.Column("source", sa.Text(), nullable=False, server_default=sa.text("'chat'")),
        sa.Column("author_requestor", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("knowledge_id"),
    )

    op.create_index(
        "ux_agent_knowledge_active_topic",
        "agent_knowledge",
        ["tenant_id", "agent_id", "topic"],
        unique=True,
        postgresql_where=sa.text("is_active = TRUE"),
    )
    op.create_index(
        "ix_agent_knowledge_tenant_agent_updated",
        "agent_knowledge",
        ["tenant_id", "agent_id", "updated_at"],
        unique=False,
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_agent_knowledge_tsv
        ON agent_knowledge
        USING gin (to_tsvector('simple', topic || ' ' || content))
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_agent_knowledge_tsv")
    op.drop_index("ix_agent_knowledge_tenant_agent_updated", table_name="agent_knowledge")
    op.drop_index("ux_agent_knowledge_active_topic", table_name="agent_knowledge")
    op.drop_table("agent_knowledge")

