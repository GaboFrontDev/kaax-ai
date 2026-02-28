from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_certificatemanager as acm,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_elasticloadbalancingv2 as elbv2,
    aws_iam as iam,
    aws_logs as logs,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


_COMPLETE_SECRET_ARN_RE = re.compile(r"^arn:[^:]+:secretsmanager:[^:]+:[0-9]{12}:secret:.+-[A-Za-z0-9]{6}$")


@dataclass(frozen=True)
class DeploymentConfig:
    account: str | None
    region: str | None
    agent: str
    service_name: str
    container_port: int = 8200
    cpu: int = 512
    memory_mib: int = 1024
    cpu_architecture: str = "X86_64"
    desired_count: int = 1
    min_capacity: int = 1
    max_capacity: int = 2
    public_load_balancer: bool = True
    enable_https: bool = False
    redirect_http: bool = True
    certificate_arn: str | None = None
    public_base_url: str | None = None
    health_check_path: str = "/health/live"
    health_check_grace_seconds: int = 120
    vpc_id: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    secret_name: str | None = None
    secret_arn: str | None = None
    secret_keys: list[str] = field(default_factory=list)


def load_deployment_config(
    *,
    config_file: str,
    env_name: str,
    agent_name: str,
) -> DeploymentConfig:
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parents[1] / config_path

    if not config_path.exists():
        raise ValueError(
            f"Config file not found: {config_path}. "
            "Create it from infra/cdk/config/environments.example.json"
        )

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    env_raw = raw.get(env_name)
    if not isinstance(env_raw, dict):
        available = ", ".join(sorted(raw.keys()))
        raise ValueError(f"Environment '{env_name}' not found. Available: {available}")

    agents_raw = env_raw.get("agents")
    if not isinstance(agents_raw, dict):
        raise ValueError(f"Environment '{env_name}' has no 'agents' map")

    agent_raw = agents_raw.get(agent_name)
    if not isinstance(agent_raw, dict):
        available_agents = ", ".join(sorted(agents_raw.keys()))
        raise ValueError(
            f"Agent '{agent_name}' not found in '{env_name}'. Available: {available_agents}"
        )

    return DeploymentConfig(
        account=env_raw.get("account"),
        region=env_raw.get("region"),
        agent=agent_name,
        service_name=str(agent_raw.get("service_name", f"kaax-{env_name}-{agent_name}")),
        container_port=int(agent_raw.get("container_port", 8200)),
        cpu=int(agent_raw.get("cpu", 512)),
        memory_mib=int(agent_raw.get("memory_mib", 1024)),
        cpu_architecture=str(agent_raw.get("cpu_architecture", "X86_64")).upper(),
        desired_count=int(agent_raw.get("desired_count", 1)),
        min_capacity=int(agent_raw.get("min_capacity", 1)),
        max_capacity=int(agent_raw.get("max_capacity", 2)),
        public_load_balancer=bool(agent_raw.get("public_load_balancer", True)),
        enable_https=bool(agent_raw.get("enable_https", False)),
        redirect_http=bool(agent_raw.get("redirect_http", True)),
        certificate_arn=agent_raw.get("certificate_arn"),
        public_base_url=agent_raw.get("public_base_url"),
        health_check_path=str(agent_raw.get("health_check_path", "/health/live")),
        health_check_grace_seconds=int(agent_raw.get("health_check_grace_seconds", 120)),
        vpc_id=agent_raw.get("vpc_id"),
        environment={
            str(key): str(value)
            for key, value in (agent_raw.get("environment", {}) or {}).items()
        },
        secret_name=agent_raw.get("secret_name"),
        secret_arn=agent_raw.get("secret_arn"),
        secret_keys=[str(key) for key in (agent_raw.get("secret_keys", []) or [])],
    )


class KaaxServiceStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: DeploymentConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if config.vpc_id:
            vpc = ec2.Vpc.from_lookup(self, "Vpc", vpc_id=config.vpc_id)
        else:
            vpc = ec2.Vpc(
                self,
                "Vpc",
                nat_gateways=1,
                max_azs=2,
            )

        cluster = ecs.Cluster(
            self,
            "Cluster",
            vpc=vpc,
            cluster_name=f"{config.service_name}-cluster",
        )

        arch = config.cpu_architecture.strip().upper()
        if arch == "ARM64":
            task_cpu_arch = ecs.CpuArchitecture.ARM64
            asset_platform = ecr_assets.Platform.LINUX_ARM64
        else:
            task_cpu_arch = ecs.CpuArchitecture.X86_64
            asset_platform = ecr_assets.Platform.LINUX_AMD64

        if config.enable_https and not config.certificate_arn:
            raise ValueError(
                "HTTPS enabled but certificate_arn is missing in deployment config"
            )

        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDefinition",
            cpu=config.cpu,
            memory_limit_mib=config.memory_mib,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=task_cpu_arch,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )

        log_group = logs.LogGroup(
            self,
            "AppLogs",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        secrets: dict[str, ecs.Secret] = {}
        if config.secret_name or config.secret_arn:
            if config.secret_name:
                secret = secretsmanager.Secret.from_secret_name_v2(
                    self,
                    "AppSecret",
                    config.secret_name,
                )
            else:
                if config.secret_arn and _COMPLETE_SECRET_ARN_RE.match(config.secret_arn):
                    secret = secretsmanager.Secret.from_secret_complete_arn(
                        self,
                        "AppSecret",
                        config.secret_arn,
                    )
                else:
                    # Fallback for partial ARN input.
                    secret = secretsmanager.Secret.from_secret_partial_arn(
                        self,
                        "AppSecret",
                        config.secret_arn,
                    )
            if config.secret_keys:
                for key in config.secret_keys:
                    secrets[key] = ecs.Secret.from_secrets_manager(secret, key)
            else:
                secrets["APP_SECRET_BUNDLE"] = ecs.Secret.from_secrets_manager(secret)
            secret.grant_read(task_definition.task_role)

        default_env = {
            "AUDRAI_DEPLOY_ENV": "aws",
            "LOG_FORMAT": "json",
            "LOG_LEVEL": "INFO",
            "AGENT_RUNTIME_STRICT": "true",
        }
        default_env.update(config.environment)

        # Runtime model ids can come from Secrets Manager, so we can't rely on synth-time env vars.
        # Grant invoke permissions broadly for Bedrock and tighten later if desired.
        task_definition.task_role.add_to_principal_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=["*"],
            )
        )

        container = task_definition.add_container(
            "AppContainer",
            image=ecs.ContainerImage.from_asset(
                directory=str(Path(__file__).resolve().parents[3]),
                file="Dockerfile",
                platform=asset_platform,
            ),
            environment=default_env,
            secrets=secrets,
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix=config.service_name,
                log_group=log_group,
            ),
        )
        container.add_port_mappings(
            ecs.PortMapping(container_port=config.container_port)
        )

        certificate = (
            acm.Certificate.from_certificate_arn(
                self,
                "AlbCertificate",
                config.certificate_arn,
            )
            if config.enable_https and config.certificate_arn
            else None
        )

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "Service",
            service_name=config.service_name,
            cluster=cluster,
            task_definition=task_definition,
            desired_count=config.desired_count,
            public_load_balancer=config.public_load_balancer,
            health_check_grace_period=Duration.seconds(
                config.health_check_grace_seconds
            ),
            listener_port=443 if config.enable_https else 80,
            protocol=(
                elbv2.ApplicationProtocol.HTTPS
                if config.enable_https
                else elbv2.ApplicationProtocol.HTTP
            ),
            certificate=certificate,
            redirect_http=(config.redirect_http if config.enable_https else False),
        )

        fargate_service.target_group.configure_health_check(
            path=config.health_check_path,
            healthy_http_codes="200",
            interval=Duration.seconds(30),
            timeout=Duration.seconds(10),
        )

        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=config.min_capacity,
            max_capacity=config.max_capacity,
        )
        scaling.scale_on_cpu_utilization(
            "CpuAutoscaling",
            target_utilization_percent=65,
            scale_in_cooldown=Duration.seconds(60),
            scale_out_cooldown=Duration.seconds(60),
        )

        CfnOutput(
            self,
            "ServiceName",
            value=fargate_service.service.service_name,
        )
        CfnOutput(
            self,
            "ClusterName",
            value=cluster.cluster_name,
        )
        CfnOutput(
            self,
            "LoadBalancerDNS",
            value=fargate_service.load_balancer.load_balancer_dns_name,
        )
        base_url = config.public_base_url
        if not base_url:
            scheme = "https" if config.enable_https else "http"
            base_url = f"{scheme}://{fargate_service.load_balancer.load_balancer_dns_name}"
        CfnOutput(
            self,
            "BaseUrl",
            value=base_url,
        )
