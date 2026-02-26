from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.agent.runtime import AgentRuntime
from app.api.dependencies import get_agent_runtime, get_slack_dlq, get_slack_message_queue
from app.channels.slack.adapter import SlackAdapter
from app.channels.slack.dlq import SlackDeadLetterQueue
from app.channels.slack.queue import SlackMessageQueue
from app.memory.session_manager import SessionBusyError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["slack"])
_adapter = SlackAdapter()


@router.post("/slack/events")
async def slack_events(
    payload: dict[str, object],
    runtime: AgentRuntime = Depends(get_agent_runtime),
    dlq: SlackDeadLetterQueue = Depends(get_slack_dlq),
    queue: SlackMessageQueue = Depends(get_slack_message_queue),
) -> dict[str, object]:
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge", "")}

    if payload.get("type") != "event_callback":
        return {"ok": True}

    req = await _adapter.normalize_inbound(payload)
    try:
        result = await runtime.invoke(req)
        return {"ok": True, "run_id": result.get("run_id")}
    except SessionBusyError:
        await queue.enqueue(req.thread_id, payload)
        return {"ok": True, "queued": True, "queue_backend": queue.backend_name()}
    except Exception as exc:  # pragma: no cover - defensive path
        await dlq.enqueue(payload, str(exc))
        logger.exception("slack_event_failed_dead_lettered")
        return {"ok": False, "dead_lettered": True}
