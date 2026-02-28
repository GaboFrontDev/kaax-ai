from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.runtime import AgentRuntime
from app.api.auth import require_bearer_auth
from app.api.dependencies import get_agent_runtime, get_idempotency_store, get_interaction_metrics_store
from app.api.schemas.assist import AgentAssistRequest, AgentAssistResponse
from app.api.sse import with_heartbeat
from app.memory.idempotency import InMemoryIdempotencyStore
from app.memory.session_manager import SessionBusyError
from app.observability.metrics import InteractionMetricsStore

router = APIRouter(prefix="/api/agent", tags=["assist"])


@router.post("/assist", response_model=AgentAssistResponse)
async def assist(
    payload: AgentAssistRequest,
    _: str = Depends(require_bearer_auth),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    idempotency_store: InMemoryIdempotencyStore = Depends(get_idempotency_store),
    metrics_store: InteractionMetricsStore = Depends(get_interaction_metrics_store),
    request_id: str | None = Header(default=None, alias="X-Request-Id"),
):
    runtime_req = payload.to_runtime_request()
    await _record_metrics_event(
        metrics_store,
        channel="api",
        user_id=payload.requestor,
        thread_id=runtime_req.thread_id,
        direction="inbound",
        event_type="assist_request",
        success=True,
        run_id=None,
        metadata={"stream_response": payload.stream_response},
    )

    if payload.stream_response:
        await _record_metrics_event(
            metrics_store,
            channel="api",
            user_id=payload.requestor,
            thread_id=runtime_req.thread_id,
            direction="outbound",
            event_type="assist_stream_started",
            success=True,
            run_id=None,
            metadata={"stream_response": True},
        )
        stream = with_heartbeat(runtime.stream(runtime_req))
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if request_id:
        status = await idempotency_store.begin(thread_id=runtime_req.thread_id, request_id=request_id)
        if status.state == "replay" and status.response is not None:
            return AgentAssistResponse(**status.response)
        if status.state == "in_progress":
            raise HTTPException(
                status_code=409,
                detail=f"duplicate request in progress for thread={runtime_req.thread_id} request_id={request_id}",
            )

    try:
        result = await runtime.invoke(runtime_req)
    except SessionBusyError as exc:
        await _record_metrics_event(
            metrics_store,
            channel="api",
            user_id=payload.requestor,
            thread_id=runtime_req.thread_id,
            direction="outbound",
            event_type="assist_session_busy",
            success=False,
            run_id=None,
            metadata={"error": str(exc)},
        )
        if request_id:
            await idempotency_store.fail(thread_id=runtime_req.thread_id, request_id=request_id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        await _record_metrics_event(
            metrics_store,
            channel="api",
            user_id=payload.requestor,
            thread_id=runtime_req.thread_id,
            direction="outbound",
            event_type="assist_error",
            success=False,
            run_id=None,
            metadata={"error": "runtime_exception"},
        )
        if request_id:
            await idempotency_store.fail(thread_id=runtime_req.thread_id, request_id=request_id)
        raise

    if request_id:
        await idempotency_store.complete(
            thread_id=runtime_req.thread_id,
            request_id=request_id,
            response=result,
        )
    await _record_metrics_event(
        metrics_store,
        channel="api",
        user_id=payload.requestor,
        thread_id=runtime_req.thread_id,
        direction="outbound",
        event_type="assist_response",
        success=True,
        run_id=str(result.get("run_id") or ""),
        metadata={
            "tools_used": result.get("tools_used", []),
            "memory_intent": result.get("memory_intent"),
        },
    )
    return AgentAssistResponse(**result)


async def _record_metrics_event(
    metrics_store: InteractionMetricsStore,
    *,
    channel: str,
    user_id: str | None,
    thread_id: str,
    direction: str,
    event_type: str,
    success: bool,
    run_id: str | None,
    metadata: dict[str, object] | None,
) -> None:
    try:
        await metrics_store.record_event(
            channel=channel,
            user_id=user_id,
            thread_id=thread_id,
            direction=direction,
            event_type=event_type,
            success=success,
            run_id=run_id,
            metadata=metadata,
        )
    except Exception:
        # Never break runtime path because of observability failures.
        pass
