from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    get_agent_runtime,
    get_attachment_store,
    get_checkpoint_store,
    get_idempotency_store,
    get_langgraph_checkpointer_manager,
    get_lock_manager,
    get_slack_message_queue,
    get_session_manager,
)
from app.channels.slack.queue import SlackMessageQueue
from app.memory.attachments_store import AttachmentStore
from app.memory.checkpoint_store import CheckpointStore
from app.memory.locks import SessionLockManager

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(
    checkpoint_store: CheckpointStore = Depends(get_checkpoint_store),
    lock_manager: SessionLockManager = Depends(get_lock_manager),
    attachment_store: AttachmentStore = Depends(get_attachment_store),
    message_queue: SlackMessageQueue = Depends(get_slack_message_queue),
    _: object = Depends(get_session_manager),
    __: object = Depends(get_idempotency_store),
    ___: object = Depends(get_agent_runtime),
) -> dict[str, str]:
    checkpoints_ok = await checkpoint_store.is_healthy()
    locks_ok = await lock_manager.is_healthy()
    attachments_ok = await attachment_store.is_healthy()
    message_queue_ok = await message_queue.is_healthy()
    langgraph_checkpointer_manager = get_langgraph_checkpointer_manager()
    langgraph_checkpoint_backend = "disabled"
    langgraph_checkpoint_ok = True
    if langgraph_checkpointer_manager is not None:
        langgraph_checkpoint_backend = langgraph_checkpointer_manager.backend_name()
        langgraph_checkpoint_ok = await langgraph_checkpointer_manager.is_healthy()

    if (
        not checkpoints_ok
        or not locks_ok
        or not attachments_ok
        or not message_queue_ok
        or not langgraph_checkpoint_ok
    ):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checkpoint_backend": checkpoint_store.backend_name(),
                "lock_backend": lock_manager.backend_name(),
                "attachment_backend": attachment_store.backend_name(),
                "message_queue_backend": message_queue.backend_name(),
                "langgraph_checkpoint_backend": langgraph_checkpoint_backend,
                "checkpoints_ok": checkpoints_ok,
                "locks_ok": locks_ok,
                "attachments_ok": attachments_ok,
                "message_queue_ok": message_queue_ok,
                "langgraph_checkpoint_ok": langgraph_checkpoint_ok,
            },
        )

    return {
        "status": "ready",
        "checkpoint_backend": checkpoint_store.backend_name(),
        "lock_backend": lock_manager.backend_name(),
        "attachment_backend": attachment_store.backend_name(),
        "message_queue_backend": message_queue.backend_name(),
        "langgraph_checkpoint_backend": langgraph_checkpoint_backend,
    }
