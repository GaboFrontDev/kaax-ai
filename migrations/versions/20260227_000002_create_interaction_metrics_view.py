"""create interaction metrics 24h view

Revision ID: 20260227_000002
Revises: 20260227_000001
Create Date: 2026-02-27 00:00:02
"""
from __future__ import annotations

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260227_000002"
down_revision = "20260227_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS interaction_events (
            id BIGSERIAL PRIMARY KEY,
            event_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            channel TEXT NOT NULL,
            user_id TEXT NULL,
            thread_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            event_type TEXT NOT NULL,
            run_id TEXT NULL,
            success BOOLEAN NOT NULL DEFAULT TRUE,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_interaction_events_event_at ON interaction_events (event_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_interaction_events_channel_event_at ON interaction_events (channel, event_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_interaction_events_user_event_at ON interaction_events (user_id, event_at DESC)"
    )

    op.execute(
        """
        CREATE OR REPLACE VIEW interaction_metrics_24h AS
        WITH
        events_window AS (
            SELECT
                channel,
                user_id,
                thread_id,
                direction,
                success,
                event_at
            FROM interaction_events
            WHERE event_at >= NOW() - INTERVAL '24 hours'
        ),
        event_totals AS (
            SELECT
                COUNT(*)::BIGINT AS events,
                COUNT(*) FILTER (WHERE direction = 'inbound')::BIGINT AS inbound_messages,
                COUNT(*) FILTER (WHERE direction = 'outbound')::BIGINT AS outbound_messages,
                COUNT(*) FILTER (WHERE direction = 'outbound' AND success = FALSE)::BIGINT AS failed_outbound_messages,
                (COUNT(DISTINCT user_id) FILTER (WHERE user_id IS NOT NULL AND user_id <> ''))::BIGINT AS unique_users,
                COUNT(DISTINCT thread_id)::BIGINT AS active_threads
            FROM events_window
        ),
        channel_rows AS (
            SELECT
                channel,
                COUNT(*) FILTER (WHERE direction = 'inbound')::BIGINT AS inbound_messages,
                COUNT(*) FILTER (WHERE direction = 'outbound')::BIGINT AS outbound_messages,
                COUNT(*) FILTER (WHERE direction = 'outbound' AND success = FALSE)::BIGINT AS failed_outbound_messages,
                (COUNT(DISTINCT user_id) FILTER (WHERE user_id IS NOT NULL AND user_id <> ''))::BIGINT AS unique_users
            FROM events_window
            GROUP BY channel
        ),
        channel_json AS (
            SELECT COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'channel', channel,
                        'inbound_messages', inbound_messages,
                        'outbound_messages', outbound_messages,
                        'failed_outbound_messages', failed_outbound_messages,
                        'unique_users', unique_users
                    )
                    ORDER BY inbound_messages DESC
                ),
                '[]'::jsonb
            ) AS channels
            FROM channel_rows
        ),
        top_user_rows AS (
            SELECT
                user_id,
                COUNT(*) FILTER (WHERE direction = 'inbound')::BIGINT AS inbound_messages,
                MAX(event_at) AS last_seen,
                ARRAY_AGG(DISTINCT channel) AS channels
            FROM events_window
            WHERE user_id IS NOT NULL AND user_id <> ''
            GROUP BY user_id
            ORDER BY inbound_messages DESC, last_seen DESC
            LIMIT 20
        ),
        top_user_json AS (
            SELECT COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'user_id', user_id,
                        'inbound_messages', inbound_messages,
                        'last_seen', last_seen,
                        'channels', to_jsonb(channels)
                    )
                    ORDER BY inbound_messages DESC, last_seen DESC
                ),
                '[]'::jsonb
            ) AS top_users
            FROM top_user_rows
        ),
        lead_totals AS (
            SELECT
                COUNT(*)::BIGINT AS lead_total,
                COUNT(*) FILTER (WHERE lead_status = 'calificado')::BIGINT AS lead_qualified,
                COUNT(*) FILTER (WHERE lead_status = 'en_revision')::BIGINT AS lead_in_review,
                COUNT(*) FILTER (WHERE lead_status = 'no_calificado')::BIGINT AS lead_disqualified
            FROM crm_leads
            WHERE created_at >= NOW() - INTERVAL '24 hours'
        )
        SELECT
            NOW() AS calculated_at,
            et.events,
            et.inbound_messages,
            et.outbound_messages,
            et.failed_outbound_messages,
            et.unique_users,
            et.active_threads,
            lt.lead_total,
            lt.lead_qualified,
            lt.lead_in_review,
            lt.lead_disqualified,
            CASE
                WHEN lt.lead_total > 0 THEN (lt.lead_qualified::numeric / lt.lead_total::numeric)
                ELSE NULL
            END AS lead_qualification_rate,
            cj.channels,
            tu.top_users
        FROM event_totals et
        CROSS JOIN lead_totals lt
        CROSS JOIN channel_json cj
        CROSS JOIN top_user_json tu
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS interaction_metrics_24h")
