from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _split_csv(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _redis_sentinel_nodes_from_env() -> tuple[str, ...]:
    nodes: list[str] = []
    for index in (1, 2, 3):
        host = os.getenv(f"REDIS_SENTINEL_{index}_HOST")
        port = os.getenv(f"REDIS_SENTINEL_{index}_PORT")
        if not host:
            continue
        resolved_port = port if port else "26379"
        nodes.append(f"{host}:{resolved_port}")

    if nodes:
        return tuple(nodes)

    csv_raw = os.getenv("REDIS_SENTINELS")
    if csv_raw:
        parsed = [node.strip() for node in csv_raw.split(",") if node.strip()]
        if parsed:
            return tuple(parsed)

    return ("127.0.0.1:26379",)


@dataclass(frozen=True)
class Settings:
    api_tokens: set[str]
    deploy_env: str
    log_level: str
    log_format: str
    log_colorized: bool
    sentry_dsn: str | None
    checkpoint_backend: str
    attachment_backend: str
    message_queue_backend: str
    file_retention_count: int
    file_retention_minutes: int
    message_queue_max_size: int
    session_timeout_seconds: int
    cleanup_interval_seconds: int
    cleanup_jitter_seconds: int
    idempotency_ttl_seconds: int
    db_dsn: str | None
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str
    db_ssl_mode: str | None
    db_pool_min_size: int
    db_pool_max_size: int
    db_command_timeout_seconds: float
    crm_backend: str
    crm_table_name: str
    knowledge_backend: str
    knowledge_table_name: str
    knowledge_admin_requestors: set[str]
    knowledge_search_default_limit: int
    knowledge_learn_confidence_threshold: float
    agent_id: str
    interaction_metrics_backend: str
    interaction_metrics_table_name: str
    redis_master_name: str
    redis_password: str | None
    redis_db: int
    redis_socket_timeout_seconds: float
    redis_sentinel_nodes: tuple[str, ...]
    redis_master_host_override: str | None
    redis_master_port_override: int | None
    agent_runtime_backend: str
    agent_runtime_strict: bool
    model_name: str
    small_model_name: str
    model_temperature: float
    aws_region: str
    prompt_name: str
    langchain_summarization_enabled: bool
    lead_owner_notify_enabled: bool
    lead_owner_whatsapp_number: str | None
    whatsapp_meta_owner_phone_number_id: str | None
    whatsapp_twilio_auth_token: str | None
    whatsapp_twilio_webhook_url: str | None
    whatsapp_meta_verify_token: str | None
    whatsapp_meta_app_secret: str | None
    whatsapp_meta_access_token: str | None
    whatsapp_meta_api_version: str

    @staticmethod
    def from_env() -> "Settings":
        tokens = _split_csv(os.getenv("API_TOKENS"))
        if not tokens:
            tokens = {"dev-token"}

        return Settings(
            api_tokens=tokens,
            deploy_env=os.getenv("AUDRAI_DEPLOY_ENV", "local"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv(
                "LOG_FORMAT",
                "pretty" if os.getenv("AUDRAI_DEPLOY_ENV", "local").strip().lower() == "local" else "json",
            ),
            log_colorized=_bool_env("LOG_COLORIZED", True),
            sentry_dsn=os.getenv("SENTRY_DSN"),
            checkpoint_backend=os.getenv("CHECKPOINT_BACKEND", "memory"),
            attachment_backend=os.getenv("ATTACHMENT_BACKEND", "memory"),
            message_queue_backend=os.getenv("MESSAGE_QUEUE_BACKEND", "memory"),
            file_retention_count=int(os.getenv("FILE_RETENTION_COUNT", "20")),
            file_retention_minutes=int(os.getenv("FILE_RETENTION_MINUTES", "120")),
            message_queue_max_size=int(os.getenv("MESSAGE_QUEUE_MAX_SIZE", "200")),
            session_timeout_seconds=int(os.getenv("SESSION_TIMEOUT_SECONDS", "1800")),
            cleanup_interval_seconds=int(os.getenv("CLEANUP_INTERVAL_SECONDS", "60")),
            cleanup_jitter_seconds=int(os.getenv("CLEANUP_JITTER_SECONDS", "20")),
            idempotency_ttl_seconds=int(os.getenv("IDEMPOTENCY_TTL_SECONDS", "3600")),
            db_dsn=os.getenv("DB_DSN"),
            db_user=os.getenv("DB_USER", "postgres"),
            db_password=os.getenv("DB_PASSWORD", "postgres"),
            db_host=os.getenv("DB_HOST", "localhost"),
            db_port=int(os.getenv("DB_PORT", "5432")),
            db_name=os.getenv("DB_NAME", "postgres"),
            db_ssl_mode=os.getenv("DB_SSL_MODE"),
            db_pool_min_size=int(os.getenv("DB_POOL_MIN_SIZE", "1")),
            db_pool_max_size=int(os.getenv("DB_POOL_MAX_SIZE", "10")),
            db_command_timeout_seconds=float(os.getenv("DB_COMMAND_TIMEOUT_SECONDS", "30")),
            crm_backend=os.getenv("CRM_BACKEND", "memory"),
            crm_table_name=os.getenv("CRM_TABLE_NAME", "crm_leads"),
            knowledge_backend=os.getenv("KNOWLEDGE_BACKEND", "memory"),
            knowledge_table_name=os.getenv("KNOWLEDGE_TABLE_NAME", "agent_knowledge"),
            knowledge_admin_requestors=_split_csv(os.getenv("KNOWLEDGE_ADMIN_REQUESTORS")),
            knowledge_search_default_limit=int(os.getenv("KNOWLEDGE_SEARCH_DEFAULT_LIMIT", "5")),
            knowledge_learn_confidence_threshold=float(os.getenv("KNOWLEDGE_LEARN_CONFIDENCE_THRESHOLD", "0.75")),
            agent_id=os.getenv("AGENT_ID", "default"),
            interaction_metrics_backend=os.getenv("INTERACTION_METRICS_BACKEND", "auto"),
            interaction_metrics_table_name=os.getenv("INTERACTION_METRICS_TABLE", "interaction_events"),
            redis_master_name=os.getenv("REDIS_MASTER_NAME", "mymaster"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_socket_timeout_seconds=float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "1.0")),
            redis_sentinel_nodes=_redis_sentinel_nodes_from_env(),
            redis_master_host_override=os.getenv("REDIS_MASTER_HOST_OVERRIDE"),
            redis_master_port_override=(
                int(os.getenv("REDIS_MASTER_PORT_OVERRIDE"))
                if os.getenv("REDIS_MASTER_PORT_OVERRIDE")
                else None
            ),
            agent_runtime_backend=os.getenv("AGENT_RUNTIME_BACKEND", "langgraph_mvp"),
            agent_runtime_strict=_bool_env("AGENT_RUNTIME_STRICT", False),
            model_name=os.getenv(
                "MODEL_NAME",
                os.getenv("SONNET_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
            ),
            small_model_name=os.getenv(
                "SMALL_MODEL",
                os.getenv("SONNET_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"),
            ),
            model_temperature=float(os.getenv("MODEL_TEMPERATURE", "0.5")),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            prompt_name=os.getenv("PROMPT_NAME", "agent"),
            langchain_summarization_enabled=_bool_env("LANGCHAIN_SUMMARIZATION_ENABLED", True),
            lead_owner_notify_enabled=_bool_env("LEAD_OWNER_NOTIFY_ENABLED", False),
            lead_owner_whatsapp_number=os.getenv("LEAD_OWNER_WHATSAPP_NUMBER"),
            whatsapp_meta_owner_phone_number_id=os.getenv("WHATSAPP_META_OWNER_PHONE_NUMBER_ID"),
            whatsapp_twilio_auth_token=os.getenv("WHATSAPP_TWILIO_AUTH_TOKEN"),
            whatsapp_twilio_webhook_url=os.getenv("WHATSAPP_TWILIO_WEBHOOK_URL"),
            whatsapp_meta_verify_token=os.getenv("WHATSAPP_META_VERIFY_TOKEN"),
            whatsapp_meta_app_secret=os.getenv("WHATSAPP_META_APP_SECRET"),
            whatsapp_meta_access_token=os.getenv("WHATSAPP_META_ACCESS_TOKEN"),
            whatsapp_meta_api_version=os.getenv("WHATSAPP_META_API_VERSION", "v21.0"),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
