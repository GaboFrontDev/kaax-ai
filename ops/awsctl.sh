#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_ENV="${ENV:-dev}"
DEFAULT_AGENT="${AGENT:-default}"
DEFAULT_REGION="${AWS_REGION:-us-east-1}"

print_section() {
  echo
  echo "== $1 =="
}

pass() {
  echo "[OK] $1"
}

warn() {
  echo "[WARN] $1"
}

fail() {
  echo "[FAIL] $1"
}

stack_name() {
  local env_name="$1"
  local agent_name="$2"
  echo "Kaax-${env_name}-${agent_name}"
}

default_service_name() {
  local env_name="$1"
  local agent_name="$2"
  echo "kaax-${env_name}-${agent_name}"
}

default_cluster_name() {
  local env_name="$1"
  local agent_name="$2"
  local service_name
  service_name="$(default_service_name "$env_name" "$agent_name")"
  echo "${service_name}-cluster"
}

stack_output() {
  local stack="$1"
  local output_key="$2"
  aws cloudformation describe-stacks \
    --stack-name "$stack" \
    --region "$DEFAULT_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='${output_key}'].OutputValue | [0]" \
    --output text
}

stack_output_safe() {
  local stack="$1"
  local output_key="$2"
  aws cloudformation describe-stacks \
    --stack-name "$stack" \
    --region "$DEFAULT_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='${output_key}'].OutputValue | [0]" \
    --output text 2>/dev/null || true
}

resolve_base_url() {
  local stack="$1"
  local base_url
  local lb_dns
  base_url="$(stack_output_safe "$stack" "BaseUrl")"
  if [[ -n "$base_url" && "$base_url" != "None" ]]; then
    echo "$base_url"
    return
  fi
  lb_dns="$(stack_output_safe "$stack" "LoadBalancerDNS")"
  if [[ -n "$lb_dns" && "$lb_dns" != "None" ]]; then
    echo "http://${lb_dns}"
    return
  fi
  echo ""
}

resolve_service_name() {
  local stack="$1"
  local env_name="$2"
  local agent_name="$3"
  local service_name
  service_name="$(stack_output_safe "$stack" "ServiceName")"
  if [[ -z "$service_name" || "$service_name" == "None" ]]; then
    service_name="$(default_service_name "$env_name" "$agent_name")"
  fi
  echo "$service_name"
}

resolve_cluster_name() {
  local stack="$1"
  local env_name="$2"
  local agent_name="$3"
  local cluster_name
  cluster_name="$(stack_output_safe "$stack" "ClusterName")"
  if [[ -z "$cluster_name" || "$cluster_name" == "None" ]]; then
    cluster_name="$(default_cluster_name "$env_name" "$agent_name")"
  fi
  echo "$cluster_name"
}

cluster_exists() {
  local cluster_name="$1"
  local status
  status="$(
    aws ecs describe-clusters \
      --clusters "$cluster_name" \
      --region "$DEFAULT_REGION" \
      --query "clusters[0].status" \
      --output text 2>/dev/null || true
  )"
  [[ -n "$status" && "$status" != "None" && "$status" != "INACTIVE" ]]
}

usage() {
  cat <<'EOF'
Usage:
  ./ops/awsctl.sh <command> [env] [agent]

Commands:
  deploy [env] [agent]        Run CDK deploy
  diff [env] [agent]          Run CDK diff
  destroy [env] [agent]       Run CDK destroy
  sync-secrets [secret-name]  Sync shell exports to AWS Secrets Manager
  status [env] [agent]        CloudFormation stack status
  events [env] [agent]        CloudFormation stack events (latest 25)
  ecs-events [env] [agent]    ECS service events (latest 20)
  task-status [env] [agent]   ECS running/pending/desired/deployment status
  task-fail [env] [agent]     Last stopped task failure reason
  doctor [env] [agent]        One-shot diagnostics (stack+ecs+logs+health)
  logs [env] [agent] [since]  Tail CloudWatch logs (default since=15m)
  lb [env] [agent]            Print load balancer DNS
  health [env] [agent]        Curl /health/live via load balancer
  help                        Show this help

Defaults:
  env=dev, agent=default, region from AWS_REGION (fallback us-east-1)
EOF
}

command_name="${1:-help}"
case "$command_name" in
  deploy)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    exec "$ROOT_DIR/ops/deploy.sh" "$env_name" "$agent_name"
    ;;
  diff)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    exec "$ROOT_DIR/ops/diff.sh" "$env_name" "$agent_name"
    ;;
  destroy)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    exec "$ROOT_DIR/ops/destroy.sh" "$env_name" "$agent_name"
    ;;
  sync-secrets)
    secret_name="${2:-}"
    exec "$ROOT_DIR/ops/secrets-sync.sh" "$secret_name"
    ;;
  status)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"
    aws cloudformation describe-stacks \
      --stack-name "$stack" \
      --region "$DEFAULT_REGION" \
      --query "Stacks[0].[StackStatus,StackStatusReason]" \
      --output table
    ;;
  events)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"
    aws cloudformation describe-stack-events \
      --stack-name "$stack" \
      --region "$DEFAULT_REGION" \
      --max-items 25 \
      --query "StackEvents[].[Timestamp,LogicalResourceId,ResourceStatus,ResourceStatusReason]" \
      --output table
    ;;
  ecs-events)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"
    cluster_name="$(resolve_cluster_name "$stack" "$env_name" "$agent_name")"
    service_name="$(resolve_service_name "$stack" "$env_name" "$agent_name")"
    if ! cluster_exists "$cluster_name"; then
      warn "Cluster not found yet: ${cluster_name}"
      warn "Stack may still be creating resources. Try 'status' or 'events' first."
      exit 2
    fi
    aws ecs describe-services \
      --cluster "$cluster_name" \
      --services "$service_name" \
      --region "$DEFAULT_REGION" \
      --query "services[0].events[0:20].[createdAt,message]" \
      --output table
    ;;
  task-status)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"
    cluster_name="$(resolve_cluster_name "$stack" "$env_name" "$agent_name")"
    service_name="$(resolve_service_name "$stack" "$env_name" "$agent_name")"
    if ! cluster_exists "$cluster_name"; then
      warn "Cluster not found yet: ${cluster_name}"
      warn "Stack may still be creating resources. Try again in a minute."
      exit 2
    fi
    aws ecs describe-services \
      --cluster "$cluster_name" \
      --services "$service_name" \
      --region "$DEFAULT_REGION" \
      --query "services[0].[runningCount,pendingCount,desiredCount,deployments[0].rolloutState]" \
      --output table
    ;;
  task-fail)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"
    cluster_name="$(resolve_cluster_name "$stack" "$env_name" "$agent_name")"
    service_name="$(resolve_service_name "$stack" "$env_name" "$agent_name")"
    if ! cluster_exists "$cluster_name"; then
      warn "Cluster not found yet: ${cluster_name}"
      warn "No stopped task details available yet."
      exit 2
    fi
    task_arn="$(
      aws ecs list-tasks \
        --cluster "$cluster_name" \
        --service-name "$service_name" \
        --desired-status STOPPED \
        --region "$DEFAULT_REGION" \
        --query "taskArns[0]" \
        --output text
    )"
    if [[ -z "$task_arn" || "$task_arn" == "None" ]]; then
      echo "No stopped tasks found."
      exit 0
    fi
    aws ecs describe-tasks \
      --cluster "$cluster_name" \
      --tasks "$task_arn" \
      --region "$DEFAULT_REGION" \
      --query "tasks[0].{stop:stoppedReason,containers:containers[*].{name:name,reason:reason,exit:exitCode}}" \
      --output json
    ;;
  doctor)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"

    print_section "Context"
    echo "stack=${stack}"
    echo "region=${DEFAULT_REGION}"

    print_section "AWS Identity"
    account_id="$(
      aws sts get-caller-identity --query Account --output text 2>/dev/null || true
    )"
    if [[ -n "$account_id" && "$account_id" != "None" ]]; then
      pass "Authenticated in account ${account_id}"
    else
      fail "AWS auth failed. Check credentials/profile."
      exit 1
    fi

    print_section "CloudFormation"
    stack_status="$(
      aws cloudformation describe-stacks \
        --stack-name "$stack" \
        --region "$DEFAULT_REGION" \
        --query "Stacks[0].StackStatus" \
        --output text 2>/dev/null || true
    )"
    if [[ -z "$stack_status" || "$stack_status" == "None" ]]; then
      fail "Stack not found: ${stack}"
      exit 1
    fi
    echo "status=${stack_status}"
    aws cloudformation describe-stack-events \
      --stack-name "$stack" \
      --region "$DEFAULT_REGION" \
      --max-items 8 \
      --query "StackEvents[].[Timestamp,LogicalResourceId,ResourceStatus,ResourceStatusReason]" \
      --output table || true

    cluster_name="$(resolve_cluster_name "$stack" "$env_name" "$agent_name")"
    service_name="$(resolve_service_name "$stack" "$env_name" "$agent_name")"
    lb_dns="$(stack_output_safe "$stack" "LoadBalancerDNS")"
    base_url="$(resolve_base_url "$stack")"

    print_section "Outputs"
    echo "cluster=${cluster_name:-N/A}"
    echo "service=${service_name:-N/A}"
    echo "lb_dns=${lb_dns:-N/A}"
    echo "base_url=${base_url:-N/A}"

    if cluster_exists "$cluster_name" && [[ -n "$service_name" && "$service_name" != "None" ]]; then
      print_section "ECS Service"
      aws ecs describe-services \
        --cluster "$cluster_name" \
        --services "$service_name" \
        --region "$DEFAULT_REGION" \
        --query "services[0].[runningCount,pendingCount,desiredCount,deployments[0].rolloutState]" \
        --output table || true

      print_section "ECS Events (latest)"
      aws ecs describe-services \
        --cluster "$cluster_name" \
        --services "$service_name" \
        --region "$DEFAULT_REGION" \
        --query "services[0].events[0:8].[createdAt,message]" \
        --output table || true

      task_arn="$(
        aws ecs list-tasks \
          --cluster "$cluster_name" \
          --service-name "$service_name" \
          --desired-status STOPPED \
          --region "$DEFAULT_REGION" \
          --query "taskArns[0]" \
          --output text 2>/dev/null || true
      )"
      if [[ -n "$task_arn" && "$task_arn" != "None" ]]; then
        print_section "Last Stopped Task"
        aws ecs describe-tasks \
          --cluster "$cluster_name" \
          --tasks "$task_arn" \
          --region "$DEFAULT_REGION" \
          --query "tasks[0].{stop:stoppedReason,containers:containers[*].{name:name,reason:reason,exit:exitCode}}" \
          --output json || true
      fi
    else
      warn "Cluster/Service not available yet (still provisioning)."
    fi

    if [[ -n "$base_url" ]]; then
      print_section "Healthcheck"
      http_code="$(
        curl -sS -m 8 -o /tmp/kaax_health_body.txt -w "%{http_code}" "${base_url}/health/live" || true
      )"
      if [[ "$http_code" == "200" ]]; then
        pass "GET ${base_url}/health/live -> 200"
      else
        warn "GET ${base_url}/health/live -> ${http_code:-N/A}"
        if [[ -f /tmp/kaax_health_body.txt ]]; then
          head -c 300 /tmp/kaax_health_body.txt || true
          echo
        fi
      fi
      rm -f /tmp/kaax_health_body.txt
    else
      warn "Base URL not available yet."
    fi

    print_section "Secret Check (optional)"
    if [[ -n "${CDK_SECRET_NAME:-}" ]]; then
      if aws secretsmanager describe-secret --secret-id "$CDK_SECRET_NAME" --region "$DEFAULT_REGION" >/dev/null 2>&1; then
        pass "Secret exists by name: ${CDK_SECRET_NAME}"
      else
        fail "Secret missing by name: ${CDK_SECRET_NAME}"
      fi
    elif [[ -n "${CDK_SECRET_ARN:-}" ]]; then
      if aws secretsmanager describe-secret --secret-id "$CDK_SECRET_ARN" --region "$DEFAULT_REGION" >/dev/null 2>&1; then
        pass "Secret exists by ARN."
      else
        fail "Secret missing by ARN."
      fi
    else
      warn "Set CDK_SECRET_NAME or CDK_SECRET_ARN to include secret validation."
    fi
    ;;
  logs)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    since="${4:-15m}"
    stack="$(stack_name "$env_name" "$agent_name")"
    cluster_name="$(resolve_cluster_name "$stack" "$env_name" "$agent_name")"
    service_name="$(resolve_service_name "$stack" "$env_name" "$agent_name")"

    log_group=""

    # Preferred: resolve log group from the service's current task definition.
    if cluster_exists "$cluster_name"; then
      task_definition_arn="$(
        aws ecs describe-services \
          --cluster "$cluster_name" \
          --services "$service_name" \
          --region "$DEFAULT_REGION" \
          --query "services[0].deployments[0].taskDefinition" \
          --output text 2>/dev/null || true
      )"
      if [[ -n "$task_definition_arn" && "$task_definition_arn" != "None" ]]; then
        log_group="$(
          aws ecs describe-task-definition \
            --task-definition "$task_definition_arn" \
            --region "$DEFAULT_REGION" \
            --query "taskDefinition.containerDefinitions[0].logConfiguration.options.\"awslogs-group\"" \
            --output text 2>/dev/null || true
        )"
      fi
    fi

    # Fallback: CloudFormation logical resource.
    if [[ -z "$log_group" || "$log_group" == "None" ]]; then
      log_group="$(
        aws cloudformation describe-stack-resources \
          --stack-name "$stack" \
          --region "$DEFAULT_REGION" \
          --logical-resource-id AppLogs \
          --query "StackResources[0].PhysicalResourceId" \
          --output text 2>/dev/null || true
      )"
    fi

    if [[ -z "$log_group" || "$log_group" == "None" ]]; then
      warn "No log group resolved yet. Stack/service may still be provisioning."
      warn "Try: ./ops/awsctl.sh events ${env_name} ${agent_name}"
      warn "Try: ./ops/awsctl.sh task-fail ${env_name} ${agent_name}"
      exit 2
    fi

    if ! aws logs describe-log-groups --region "$DEFAULT_REGION" --log-group-name-prefix "$log_group" --query "length(logGroups[?logGroupName=='${log_group}'])" --output text | grep -q "^1$"; then
      warn "Log group not found yet: ${log_group}"
      warn "Try again in a minute. You can still inspect ECS events with:"
      warn "./ops/awsctl.sh ecs-events ${env_name} ${agent_name}"
      exit 2
    fi

    exec aws logs tail "$log_group" --region "$DEFAULT_REGION" --since "$since" --follow
    ;;
  lb)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"
    stack_output "$stack" "LoadBalancerDNS"
    ;;
  health)
    env_name="${2:-$DEFAULT_ENV}"
    agent_name="${3:-$DEFAULT_AGENT}"
    stack="$(stack_name "$env_name" "$agent_name")"
    base_url="$(resolve_base_url "$stack")"
    if [[ -z "$base_url" ]]; then
      fail "No BaseUrl/LoadBalancerDNS output found for stack ${stack}"
      exit 2
    fi
    exec curl -i "${base_url}/health/live"
    ;;
  help | --help | -h)
    usage
    ;;
  *)
    echo "Unknown command: $command_name" >&2
    echo >&2
    usage
    exit 1
    ;;
esac
