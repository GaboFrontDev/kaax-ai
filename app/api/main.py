from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.dependencies import (
    get_attachment_store,
    get_checkpoint_store,
    get_cleanup_worker,
    get_langgraph_checkpointer_manager,
    get_postgres_pool_manager,
    get_redis_manager,
)
from app.api.routers.assist import router as assist_router
from app.api.routers.feedback import router as feedback_router
from app.api.routers.health import router as health_router
from app.api.routers.slack_events import router as slack_router
from app.api.routers.whatsapp_meta import router as whatsapp_meta_router
from app.api.routers.whatsapp_twilio import router as whatsapp_twilio_router
from app.infra.settings import get_settings
from app.observability.logging import configure_logging
from app.observability.sentry import configure_sentry


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_sentry(settings.sentry_dsn)

    checkpoint_store = get_checkpoint_store()
    await checkpoint_store.setup()

    langgraph_checkpointer_manager = get_langgraph_checkpointer_manager()
    if langgraph_checkpointer_manager is not None:
        await langgraph_checkpointer_manager.get_checkpointer()

    attachment_store = get_attachment_store()
    await attachment_store.setup()

    cleanup_worker = get_cleanup_worker()
    await cleanup_worker.start()

    try:
        yield
    finally:
        await cleanup_worker.stop()
        pool_manager = get_postgres_pool_manager()
        if pool_manager is not None:
            await pool_manager.close()
        langgraph_checkpointer_manager = get_langgraph_checkpointer_manager()
        if langgraph_checkpointer_manager is not None:
            await langgraph_checkpointer_manager.close()
        redis_manager = get_redis_manager()
        if redis_manager is not None:
            await redis_manager.close()


app = FastAPI(title="kaax-ai", version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(assist_router)
app.include_router(feedback_router)
app.include_router(slack_router)
app.include_router(whatsapp_twilio_router)
app.include_router(whatsapp_meta_router)
