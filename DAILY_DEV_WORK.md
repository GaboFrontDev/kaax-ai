# Daily Dev Work (kaax-ai)

## 1. Inicio del dia (5-10 min)

```bash
cd /Users/gabriel/repos/mine/kaax-ai
source .env.local
git status
```

Checklist rapido:
- Validar `AWS_REGION` y credenciales activas.
- Confirmar que `CDK_SECRET_NAME` apunta al entorno correcto (`kaax/dev/default`).
- Revisar estado del servicio:

```bash
make awsctl AWSCTL_ARGS="doctor dev default"
```

---

## 2. Desarrollo local

API local:

```bash
make run-api-bedrock
```

UI local (opcional):

```bash
make run-chainlit
```

Tests rapidos antes de tocar infra:

```bash
make unit-tests
```

---

## 3. Flujo de cambios en AWS (dev)

### 3.1 Sincronizar secrets desde shell

```bash
export CDK_SECRET_NAME=kaax/dev/default
export CDK_SECRET_KEYS=API_TOKENS,DB_DSN,AWS_REGION,MODEL_NAME,SMALL_MODEL,WHATSAPP_META_VERIFY_TOKEN,WHATSAPP_META_APP_SECRET,WHATSAPP_META_ACCESS_TOKEN
make cdk-sync-secrets
```

### 3.2 Ver impacto de infraestructura

```bash
make cdk-diff ENV=dev AGENT=default
```

### 3.3 Deploy

```bash
make cdk-deploy ENV=dev AGENT=default
```

### 3.4 Verificacion post-deploy

```bash
make awsctl AWSCTL_ARGS="doctor dev default"
make awsctl AWSCTL_ARGS="health dev default"
```

---

## 4. Smoke test funcional

```bash
BASE_URL=$(make awsctl AWSCTL_ARGS="lb dev default" -s)
curl -sS -X POST "http://$BASE_URL/api/agent/assist" \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"userText":"dame el iso de mexico","requestor":"smoke","streamResponse":false}'
```

Si usas dominio HTTPS:

```bash
curl -i https://api.kaax.ai/health/live
```

---

## 5. Monitoreo durante pruebas

Eventos ECS:

```bash
make awsctl AWSCTL_ARGS="ecs-events dev default"
```

Estado de tareas:

```bash
make awsctl AWSCTL_ARGS="task-status dev default"
```

Ultimo fallo de tarea:

```bash
make awsctl AWSCTL_ARGS="task-fail dev default"
```

Logs app:

```bash
make awsctl AWSCTL_ARGS="logs dev default 30m"
```

---

## 6. Flujo WhatsApp (Meta)

Checklist:
- `WHATSAPP_META_VERIFY_TOKEN`, `WHATSAPP_META_APP_SECRET`, `WHATSAPP_META_ACCESS_TOKEN` en Secrets Manager.
- Webhook Meta apuntando a:
  - `https://api.kaax.ai/webhooks/whatsapp/meta`
- Campo `messages` suscrito.

Prueba E2E:
- Enviar mensaje al numero de WhatsApp Business.
- Confirmar inbound/outbound en logs.

---

## 7. Cierre del dia

```bash
git status
```

Checklist:
- Confirmar que no quedan deploys a medias (`doctor` limpio).
- Documentar cambios clave (infra, prompts, tools, canales).
- Si hubo incidentes, guardar causa raiz y comando de solucion.
