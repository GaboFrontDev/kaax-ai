# CDK deploy (ECS Fargate)

This module deploys `kaax-ai` to AWS using CDK.

## 1) Prepare config

```bash
cd infra/cdk
cp config/environments.example.json config/environments.json
```

Edit `config/environments.json`:
- AWS account/region
- per-agent service name and sizing
- `cpu_architecture` (`X86_64` default, or `ARM64` to match ARM images)
- `enable_https=true` + `certificate_arn` (ACM) if you want TLS on ALB
- optional `public_base_url` (recommended when using custom domain)
- environment variables
- `secret_name` (recommended) or `secret_arn`, and `secret_keys`

## 2) Bootstrap and deploy

From repository root:

```bash
./ops/bootstrap.sh
./ops/deploy.sh dev default
```

Deploy another agent in same env:

```bash
./ops/deploy.sh dev sales
```

Override secret mapping from shell env (no JSON edit):

```bash
export CDK_SECRET_NAME=kaax/dev/default
export CDK_SECRET_KEYS=API_TOKENS,DB_DSN,AWS_REGION,MODEL_NAME,SMALL_MODEL,WHATSAPP_META_VERIFY_TOKEN,WHATSAPP_META_APP_SECRET,WHATSAPP_META_ACCESS_TOKEN
./ops/deploy.sh dev default
```

## 3) Useful commands

```bash
./ops/diff.sh dev default
./ops/destroy.sh dev default
```

## 4) Secrets strategy

Keep runtime secrets in one Secrets Manager secret JSON per agent, for example:

```json
{
  "API_TOKENS": "token-1,token-2",
  "DB_DSN": "postgresql://...",
  "WHATSAPP_META_ACCESS_TOKEN": "...",
  "WHATSAPP_META_APP_SECRET": "...",
  "WHATSAPP_META_VERIFY_TOKEN": "..."
}
```

Then list those keys in `secret_keys` for that agent.

You can sync from your current shell exports:

```bash
export CDK_SECRET_NAME=kaax/dev/default
export CDK_SECRET_KEYS=API_TOKENS,DB_DSN,AWS_REGION,MODEL_NAME,SMALL_MODEL,WHATSAPP_META_VERIFY_TOKEN,WHATSAPP_META_APP_SECRET,WHATSAPP_META_ACCESS_TOKEN
./ops/secrets-sync.sh
```

Note:
- Prefer `secret_name` to avoid ARN suffix issues.
- If you use `secret_arn`, partial ARN is supported.
