from __future__ import annotations

import logging
from functools import lru_cache

from app.agent.factory import build_agent
from app.agent.runtime import AgentRuntime
from app.agent.tools.registry import ToolRegistry
from app.channels.slack.dlq import SlackDeadLetterQueue
from app.channels.slack.queue import InMemorySlackMessageQueue, RedisSlackMessageQueue, SlackMessageQueue
from app.infra.db import DatabaseConfig, PostgresPoolManager, build_postgres_dsn
from app.infra.redis import RedisSentinelConfig, RedisSentinelManager, parse_sentinel_nodes
from app.infra.settings import Settings, get_settings
from app.memory.attachments_store import AttachmentStore, InMemoryAttachmentStore, RedisAttachmentStore
from app.memory.checkpoint_store import CheckpointStore, InMemoryCheckpointStore, PostgresCheckpointStore
from app.memory.cleanup import CleanupWorker
from app.memory.idempotency import InMemoryIdempotencyStore
from app.memory.langgraph_checkpointer import LangGraphCheckpointerManager
from app.memory.locks import InMemorySessionLockManager, PostgresSessionLockManager, SessionLockManager
from app.memory.session_manager import SessionManager

logger = logging.getLogger(__name__)


@lru_cache
def get_postgres_pool_manager() -> PostgresPoolManager | None:
    settings = get_settings()
    if settings.checkpoint_backend.lower() != "postgres":
        return None

    try:
        import asyncpg  # noqa: F401
    except ImportError:
        logger.warning("asyncpg_not_installed_falling_back_memory")
        return None

    try:
        dsn = build_postgres_dsn(
            db_dsn=settings.db_dsn,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
            db_name=settings.db_name,
            ssl_mode=settings.db_ssl_mode,
        )
    except Exception:
        logger.exception("postgres_dsn_build_failed_falling_back_memory")
        return None

    config = DatabaseConfig(
        dsn=dsn,
        min_pool_size=settings.db_pool_min_size,
        max_pool_size=settings.db_pool_max_size,
        command_timeout_seconds=settings.db_command_timeout_seconds,
    )
    return PostgresPoolManager(config)


@lru_cache
def get_langgraph_checkpointer_manager() -> LangGraphCheckpointerManager | None:
    settings = get_settings()
    if settings.agent_runtime_backend.lower() != "langchain":
        return None
    if settings.checkpoint_backend.lower() != "postgres":
        return None

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: F401
    except ImportError as exc:
        if settings.agent_runtime_strict:
            raise RuntimeError(
                "LangGraph native checkpoint requested but langgraph postgres checkpointer is unavailable"
            ) from exc
        logger.warning("langgraph_postgres_checkpointer_unavailable_falling_back_to_no_native_checkpoint")
        return None

    dsn = build_postgres_dsn(
        db_dsn=settings.db_dsn,
        user=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
        db_name=settings.db_name,
        ssl_mode=settings.db_ssl_mode,
    )
    return LangGraphCheckpointerManager(conn_string=dsn)


@lru_cache
def get_redis_manager() -> RedisSentinelManager | None:
    settings = get_settings()
    needs_redis = (
        settings.attachment_backend.lower() == "redis"
        or settings.message_queue_backend.lower() == "redis"
    )
    if not needs_redis:
        return None

    try:
        import redis  # noqa: F401
    except ImportError:
        logger.warning("redis_not_installed_falling_back_memory")
        return None

    try:
        sentinel_nodes = parse_sentinel_nodes(settings.redis_sentinel_nodes)
    except Exception:
        logger.exception("redis_sentinel_nodes_invalid_falling_back_memory")
        return None

    config = RedisSentinelConfig(
        sentinel_nodes=sentinel_nodes,
        master_name=settings.redis_master_name,
        password=settings.redis_password,
        db=settings.redis_db,
        socket_timeout_seconds=settings.redis_socket_timeout_seconds,
        master_host_override=settings.redis_master_host_override,
        master_port_override=settings.redis_master_port_override,
    )
    return RedisSentinelManager(config)


@lru_cache
def get_checkpoint_store() -> CheckpointStore:
    settings = get_settings()
    if settings.checkpoint_backend.lower() == "postgres":
        pool_manager = get_postgres_pool_manager()
        if pool_manager is not None:
            return PostgresCheckpointStore(pool_manager)
        logger.warning("postgres_checkpoint_backend_requested_but_unavailable_falling_back_memory")
    return InMemoryCheckpointStore()


@lru_cache
def get_lock_manager() -> SessionLockManager:
    settings = get_settings()
    if settings.checkpoint_backend.lower() == "postgres":
        pool_manager = get_postgres_pool_manager()
        if pool_manager is not None:
            return PostgresSessionLockManager(pool_manager)
        logger.warning("postgres_lock_backend_requested_but_unavailable_falling_back_memory")
    return InMemorySessionLockManager()


@lru_cache
def get_attachment_store() -> AttachmentStore:
    settings = get_settings()
    if settings.attachment_backend.lower() == "redis":
        redis_manager = get_redis_manager()
        if redis_manager is not None:
            return RedisAttachmentStore(
                redis_manager,
                max_items=settings.file_retention_count,
                ttl_minutes=settings.file_retention_minutes,
            )
        logger.warning("redis_attachment_backend_requested_but_unavailable_falling_back_memory")

    return InMemoryAttachmentStore(
        max_items=settings.file_retention_count,
        ttl_minutes=settings.file_retention_minutes,
    )


@lru_cache
def get_session_manager() -> SessionManager:
    settings = get_settings()
    return SessionManager(
        checkpoint_store=get_checkpoint_store(),
        lock_manager=get_lock_manager(),
        session_timeout_seconds=settings.session_timeout_seconds,
        lock_timeout_seconds=0,
    )


@lru_cache
def get_tool_registry() -> ToolRegistry:
    settings = get_settings()
    return ToolRegistry(
        owner_notify_enabled=settings.lead_owner_notify_enabled,
        owner_whatsapp_number=settings.lead_owner_whatsapp_number,
        owner_phone_number_id=settings.whatsapp_meta_owner_phone_number_id,
        whatsapp_meta_access_token=settings.whatsapp_meta_access_token,
        whatsapp_meta_api_version=settings.whatsapp_meta_api_version,
    )


@lru_cache
def get_idempotency_store() -> InMemoryIdempotencyStore:
    settings = get_settings()
    return InMemoryIdempotencyStore(ttl_seconds=settings.idempotency_ttl_seconds)


@lru_cache
def get_slack_dlq() -> SlackDeadLetterQueue:
    settings = get_settings()
    return SlackDeadLetterQueue(max_size=settings.message_queue_max_size)


@lru_cache
def get_slack_message_queue() -> SlackMessageQueue:
    settings = get_settings()
    if settings.message_queue_backend.lower() == "redis":
        redis_manager = get_redis_manager()
        if redis_manager is not None:
            return RedisSlackMessageQueue(
                redis_manager,
                max_size=settings.message_queue_max_size,
            )
        logger.warning("redis_message_queue_backend_requested_but_unavailable_falling_back_memory")

    return InMemorySlackMessageQueue(max_size=settings.message_queue_max_size)


@lru_cache
def get_agent_runtime() -> AgentRuntime:
    settings = get_settings()
    return build_agent(
        session_manager=get_session_manager(),
        attachment_store=get_attachment_store(),
        tool_registry=get_tool_registry(),
        tool_retry_attempts=settings.tool_retry_attempts,
        tool_retry_backoff_ms=settings.tool_retry_backoff_ms,
        runtime_backend=settings.agent_runtime_backend,
        runtime_strict=settings.agent_runtime_strict,
        model_name=settings.model_name,
        small_model_name=settings.small_model_name,
        aws_region=settings.aws_region,
        model_temperature=settings.model_temperature,
        prompt_name=settings.prompt_name,
        langchain_summarization_enabled=settings.langchain_summarization_enabled,
        llm_intent_router_enabled=settings.llm_intent_router_enabled,
        llm_intent_router_confidence_threshold=settings.llm_intent_router_confidence_threshold,
        checkpointer_manager=get_langgraph_checkpointer_manager(),
    )


@lru_cache
def get_cleanup_worker() -> CleanupWorker:
    settings: Settings = get_settings()
    return CleanupWorker(
        session_manager=get_session_manager(),
        attachment_store=get_attachment_store(),
        idempotency_store=get_idempotency_store(),
        interval_seconds=settings.cleanup_interval_seconds,
        jitter_seconds=settings.cleanup_jitter_seconds,
    )
