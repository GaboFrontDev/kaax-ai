from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.api.auth import require_bearer_auth
from app.api.schemas.feedback import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(payload: FeedbackRequest, _: str = Depends(require_bearer_auth)) -> FeedbackResponse:
    logger.info(
        "feedback_received run_id=%s score=%s source=%s",
        payload.run_id,
        payload.score,
        payload.source,
    )
    return FeedbackResponse(status="ok")
