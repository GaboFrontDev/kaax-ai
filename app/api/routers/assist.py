from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.runtime import AgentRuntime
from app.api.auth import require_bearer_auth
from app.api.dependencies import get_agent_runtime, get_idempotency_store
from app.api.schemas.assist import AgentAssistRequest, AgentAssistResponse
from app.api.sse import with_heartbeat
from app.memory.idempotency import InMemoryIdempotencyStore
from app.memory.session_manager import SessionBusyError

router = APIRouter(prefix="/api/agent", tags=["assist"])


@router.post("/assist", response_model=AgentAssistResponse)
async def assist(
    payload: AgentAssistRequest,
    _: str = Depends(require_bearer_auth),
    runtime: AgentRuntime = Depends(get_agent_runtime),
    idempotency_store: InMemoryIdempotencyStore = Depends(get_idempotency_store),
    request_id: str | None = Header(default=None, alias="X-Request-Id"),
):
    runtime_req = payload.to_runtime_request()

    if payload.stream_response:
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
        if request_id:
            await idempotency_store.fail(thread_id=runtime_req.thread_id, request_id=request_id)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        if request_id:
            await idempotency_store.fail(thread_id=runtime_req.thread_id, request_id=request_id)
        raise

    if request_id:
        await idempotency_store.complete(
            thread_id=runtime_req.thread_id,
            request_id=request_id,
            response=result,
        )
    return AgentAssistResponse(**result)
