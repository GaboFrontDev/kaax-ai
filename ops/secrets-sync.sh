#!/usr/bin/env bash
set -euo pipefail

# Sync selected env vars into one AWS Secrets Manager JSON secret.
#
# Usage:
#   ./ops/secrets-sync.sh [secret-name]
#
# Required:
#   AWS credentials configured in shell/profile
#
# Optional env vars:
#   CDK_SECRET_NAME / SECRET_NAME       -> secret name (if no arg)
#   CDK_SECRET_KEYS / SECRET_KEYS       -> comma-separated keys to include
#   AWS_REGION                          -> AWS region (default us-east-1)
#   SECRET_FORCE_OVERWRITE=true         -> overwrite with empty values (default false)
#
# Defaults:
#   If keys are not provided, a safe default key set is used.

SECRET_NAME_ARG="${1:-}"
SECRET_NAME="${SECRET_NAME_ARG:-${CDK_SECRET_NAME:-${SECRET_NAME:-}}}"
REGION="${AWS_REGION:-us-east-1}"
FORCE_OVERWRITE="${SECRET_FORCE_OVERWRITE:-false}"

if [[ -z "$SECRET_NAME" ]]; then
  echo "Missing secret name. Pass arg or set CDK_SECRET_NAME/SECRET_NAME." >&2
  exit 1
fi

DEFAULT_KEYS="API_TOKENS,DB_DSN,AWS_REGION,MODEL_NAME,SMALL_MODEL,WHATSAPP_META_VERIFY_TOKEN,WHATSAPP_META_APP_SECRET,WHATSAPP_META_ACCESS_TOKEN"
RAW_KEYS="${CDK_SECRET_KEYS:-${SECRET_KEYS:-$DEFAULT_KEYS}}"

IFS=',' read -r -a KEYS <<<"$RAW_KEYS"
if [[ "${#KEYS[@]}" -eq 0 ]]; then
  echo "No secret keys configured. Set CDK_SECRET_KEYS or SECRET_KEYS." >&2
  exit 1
fi

PAYLOAD_FILE="$(mktemp)"
cleanup() {
  rm -f "$PAYLOAD_FILE"
}
trap cleanup EXIT

python3 - "$PAYLOAD_FILE" "$FORCE_OVERWRITE" "${KEYS[@]}" <<'PY'
import json
import os
import sys

output_path = sys.argv[1]
force_overwrite = sys.argv[2].strip().lower() in {"1", "true", "yes", "on"}
keys = sys.argv[3:]

payload: dict[str, str] = {}
missing: list[str] = []
for key in keys:
    key = key.strip()
    if not key:
        continue
    value = os.getenv(key)
    if value is None:
        missing.append(key)
        continue
    if value == "" and not force_overwrite:
        missing.append(key)
        continue
    payload[key] = value

if not payload:
    sys.stderr.write("No values found to sync. Check your exported env vars.\n")
    sys.exit(2)

if missing:
    sys.stderr.write(
        "Warning: skipped missing/empty keys (not synced): "
        + ", ".join(sorted(set(missing)))
        + "\n"
    )

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=True)
PY

echo "Syncing secret '${SECRET_NAME}' in region '${REGION}'..."
echo "Keys included: ${RAW_KEYS}"

if aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$REGION" >/dev/null 2>&1; then
  aws secretsmanager put-secret-value \
    --secret-id "$SECRET_NAME" \
    --secret-string "file://${PAYLOAD_FILE}" \
    --region "$REGION" >/dev/null
  echo "Updated secret: ${SECRET_NAME}"
else
  aws secretsmanager create-secret \
    --name "$SECRET_NAME" \
    --secret-string "file://${PAYLOAD_FILE}" \
    --region "$REGION" >/dev/null
  echo "Created secret: ${SECRET_NAME}"
fi
