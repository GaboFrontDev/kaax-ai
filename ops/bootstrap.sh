#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDK_DIR="$ROOT_DIR/infra/cdk"
REGION="${AWS_REGION:-us-east-1}"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI not found. Install awscli first." >&2
  exit 1
fi

if ! command -v cdk >/dev/null 2>&1; then
  echo "cdk CLI not found. Install with: npm i -g aws-cdk" >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

python3 -m venv "$CDK_DIR/.venv"
source "$CDK_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$CDK_DIR/requirements.txt"

cd "$ROOT_DIR"
cdk bootstrap "aws://${ACCOUNT_ID}/${REGION}"

echo "CDK bootstrapped for aws://${ACCOUNT_ID}/${REGION}"
