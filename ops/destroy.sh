#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dev}"
AGENT_NAME="${2:-default}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDK_DIR="$ROOT_DIR/infra/cdk"
CONFIG_FILE="${CONFIG_FILE:-config/environments.json}"

if [[ ! -d "$CDK_DIR/.venv" ]]; then
  "$ROOT_DIR/ops/bootstrap.sh"
fi

source "$CDK_DIR/.venv/bin/activate"
cd "$CDK_DIR"

cdk destroy --force \
  -c config="$CONFIG_FILE" \
  -c env="$ENV_NAME" \
  -c agent="$AGENT_NAME"
