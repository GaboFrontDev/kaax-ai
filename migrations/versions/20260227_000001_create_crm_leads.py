"""create crm_leads table

Revision ID: 20260227_000001
Revises:
Create Date: 2026-02-27 00:00:01
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260227_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "crm_leads",
        sa.Column("crm_id", sa.Text(), nullable=False),
        sa.Column("external_key", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("lead_status", sa.Text(), nullable=True),
        sa.Column(
            "qualification_evidence",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("next_action", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.PrimaryKeyConstraint("crm_id"),
    )
    op.create_index("ix_crm_leads_external_key", "crm_leads", ["external_key"], unique=False)
    op.create_index("ix_crm_leads_created_at", "crm_leads", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_crm_leads_created_at", table_name="crm_leads")
    op.drop_index("ix_crm_leads_external_key", table_name="crm_leads")
    op.drop_table("crm_leads")
