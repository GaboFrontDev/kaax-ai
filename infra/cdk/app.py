#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
import os

import aws_cdk as cdk

from stacks.kaax_service_stack import KaaxServiceStack, load_deployment_config


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _ctx_or_env(app: cdk.App, context_key: str, *env_keys: str) -> str | None:
    context_value = app.node.try_get_context(context_key)
    if context_value is not None and str(context_value).strip():
        return str(context_value).strip()
    for env_key in env_keys:
        env_value = os.getenv(env_key)
        if env_value and env_value.strip():
            return env_value.strip()
    return None


app = cdk.App()

config_file = app.node.try_get_context("config") or "config/environments.json"
env_name = app.node.try_get_context("env") or "dev"
agent_name = app.node.try_get_context("agent") or "default"

config = load_deployment_config(
    config_file=config_file,
    env_name=env_name,
    agent_name=agent_name,
)

secret_name_override = _ctx_or_env(app, "secret_name", "CDK_SECRET_NAME", "SECRET_NAME")
secret_arn_override = _ctx_or_env(app, "secret_arn", "CDK_SECRET_ARN", "SECRET_ARN")
secret_keys_override_raw = _ctx_or_env(app, "secret_keys", "CDK_SECRET_KEYS", "SECRET_KEYS")
secret_keys_override = _split_csv(secret_keys_override_raw)

if secret_name_override or secret_arn_override or secret_keys_override:
    config = replace(
        config,
        secret_name=secret_name_override if secret_name_override else config.secret_name,
        secret_arn=secret_arn_override if secret_arn_override else config.secret_arn,
        secret_keys=secret_keys_override if secret_keys_override else config.secret_keys,
    )

stack_env = cdk.Environment(
    account=config.account or os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=config.region or os.getenv("CDK_DEFAULT_REGION"),
)

KaaxServiceStack(
    app,
    f"Kaax-{env_name}-{agent_name}",
    config=config,
    env=stack_env,
)

app.synth()
