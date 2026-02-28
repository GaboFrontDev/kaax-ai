#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config-json-file>" >&2
  echo "Required env vars: APPCONFIG_APPLICATION_ID, APPCONFIG_ENVIRONMENT_ID, APPCONFIG_PROFILE_ID" >&2
  exit 1
fi

CONFIG_JSON_FILE="$1"

if [[ ! -f "$CONFIG_JSON_FILE" ]]; then
  echo "Config file not found: $CONFIG_JSON_FILE" >&2
  exit 1
fi

: "${APPCONFIG_APPLICATION_ID:?Missing APPCONFIG_APPLICATION_ID}"
: "${APPCONFIG_ENVIRONMENT_ID:?Missing APPCONFIG_ENVIRONMENT_ID}"
: "${APPCONFIG_PROFILE_ID:?Missing APPCONFIG_PROFILE_ID}"

VERSION_NUMBER="$(aws appconfig create-hosted-configuration-version \
  --application-id "$APPCONFIG_APPLICATION_ID" \
  --configuration-profile-id "$APPCONFIG_PROFILE_ID" \
  --content-type "application/json" \
  --content "fileb://${CONFIG_JSON_FILE}" \
  --query "VersionNumber" \
  --output text)"

DEPLOYMENT_NUMBER="$(aws appconfig start-deployment \
  --application-id "$APPCONFIG_APPLICATION_ID" \
  --environment-id "$APPCONFIG_ENVIRONMENT_ID" \
  --configuration-profile-id "$APPCONFIG_PROFILE_ID" \
  --configuration-version "$VERSION_NUMBER" \
  --deployment-strategy-id "AppConfig.AllAtOnce" \
  --query "DeploymentNumber" \
  --output text)"

echo "Published AppConfig version=${VERSION_NUMBER} deployment=${DEPLOYMENT_NUMBER}"
