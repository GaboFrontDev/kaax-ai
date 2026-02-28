#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dev}"
AGENT_NAME="${2:-default}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDK_DIR="$ROOT_DIR/infra/cdk"
CONFIG_FILE="${CONFIG_FILE:-config/environments.json}"

if ! command -v cdk >/dev/null 2>&1; then
  echo "cdk CLI not found. Install with: npm i -g aws-cdk" >&2
  exit 1
fi

if [[ ! -f "$CDK_DIR/$CONFIG_FILE" ]]; then
  echo "Missing $CDK_DIR/$CONFIG_FILE" >&2
  echo "Create it from: $CDK_DIR/config/environments.example.json" >&2
  exit 1
fi

if [[ ! -d "$CDK_DIR/.venv" ]]; then
  "$ROOT_DIR/ops/bootstrap.sh"
fi

source "$CDK_DIR/.venv/bin/activate"
cd "$CDK_DIR"

EXTRA_CONTEXT_ARGS=()
if [[ -n "${CDK_SECRET_NAME:-}" ]]; then
  EXTRA_CONTEXT_ARGS+=(-c "secret_name=${CDK_SECRET_NAME}")
fi
if [[ -n "${CDK_SECRET_ARN:-}" ]]; then
  EXTRA_CONTEXT_ARGS+=(-c "secret_arn=${CDK_SECRET_ARN}")
fi
if [[ -n "${CDK_SECRET_KEYS:-}" ]]; then
  EXTRA_CONTEXT_ARGS+=(-c "secret_keys=${CDK_SECRET_KEYS}")
fi

cdk deploy \
  --require-approval never \
  -c config="$CONFIG_FILE" \
  -c env="$ENV_NAME" \
  -c agent="$AGENT_NAME" \
  "${EXTRA_CONTEXT_ARGS[@]}"
