# kaax-ai

Scaffold técnico para un agente FastAPI con memoria por `thread_id`, modo sync/stream (SSE), autenticación Bearer y estructura por capas.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
make run-api
```

Para backend de checkpoints/locks en PostgreSQL:

```bash
pip install -e .[dev,postgres]
export CHECKPOINT_BACKEND=postgres
export DB_HOST=127.0.0.1
export DB_PORT=55432
export DB_USER=postgres
export DB_PASSWORD=postgres
export DB_NAME=postgres
```

Para backend de adjuntos/colas en Redis Sentinel:

```bash
pip install -e .[dev,redis]
export ATTACHMENT_BACKEND=redis
export MESSAGE_QUEUE_BACKEND=redis
export REDIS_MASTER_NAME=mymaster
export REDIS_SENTINELS=127.0.0.1:56379,127.0.0.1:56380,127.0.0.1:56381
export REDIS_MASTER_HOST_OVERRIDE=127.0.0.1
export REDIS_MASTER_PORT_OVERRIDE=56378
```

Para runtime `create_agent` con AWS Bedrock:

```bash
pip install -e .[dev,bedrock]
export AGENT_RUNTIME_BACKEND=langchain
export AGENT_RUNTIME_STRICT=true
export AWS_REGION=us-east-1
export MODEL_NAME=anthropic.claude-3-5-sonnet-20241022-v2:0
export SMALL_MODEL=anthropic.claude-3-haiku-20240307-v1:0
export MODEL_TEMPERATURE=0.5
```

Para UI local con Chainlit:

```bash
pip install -e .[dev,chainlit]
export CHAINLIT_API_URL=http://127.0.0.1:8200
export CHAINLIT_API_TOKEN=dev-token
make run-chainlit
```

Para canal WhatsApp via Twilio (inbound + respuesta automática):

```bash
# opcional pero recomendado (firma webhook)
export WHATSAPP_TWILIO_AUTH_TOKEN=<your_twilio_auth_token>

# usa esta URL publica exacta en el sandbox de Twilio
# si usas ngrok:
export WHATSAPP_TWILIO_WEBHOOK_URL=https://<your-ngrok-subdomain>.ngrok-free.app/webhooks/whatsapp/twilio
```

Para canal WhatsApp via Meta Cloud API (inbound + respuesta automática):

```bash
export WHATSAPP_META_VERIFY_TOKEN=<your_verify_token>
export WHATSAPP_META_APP_SECRET=<your_meta_app_secret>   # recomendado
export WHATSAPP_META_ACCESS_TOKEN=<your_meta_access_token>
export WHATSAPP_META_API_VERSION=v21.0
```

## Endpoints

- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `POST /api/agent/assist`
- `POST /api/agent/feedback`
- `POST /slack/events`
- `POST /webhooks/whatsapp/twilio`
- `GET /webhooks/whatsapp/meta`
- `POST /webhooks/whatsapp/meta`

## Idempotency

- Header opcional: `X-Request-Id` en `POST /api/agent/assist` (modo sync).
- Si el request ya se procesó para el mismo `thread_id`, devuelve replay del resultado previo.

## Backends de memoria

- `CHECKPOINT_BACKEND=memory` (default): checkpoints y locks en memoria.
- `CHECKPOINT_BACKEND=postgres`: usa PostgreSQL para checkpoints y advisory locks.
- Si `CHECKPOINT_BACKEND=postgres` no puede inicializarse correctamente, el scaffold degrada a `memory`.
- `ATTACHMENT_BACKEND=memory` (default): adjuntos en memoria.
- `ATTACHMENT_BACKEND=redis`: adjuntos por thread en Redis Sentinel (TTL + límite por thread).
- `MESSAGE_QUEUE_BACKEND=memory` (default): cola Slack en memoria.
- `MESSAGE_QUEUE_BACKEND=redis`: cola Slack + thread tracking en Redis Sentinel.

## Runtime de agente

- `AGENT_RUNTIME_BACKEND=stub` (default): runtime interno deterministico para scaffolding.
- `AGENT_RUNTIME_BACKEND=langchain`: usa `langchain.agents.create_agent` con `ChatBedrockConverse`.
- `AGENT_RUNTIME_STRICT=true`: falla startup si Bedrock/LangChain no inicializa (sin fallback silencioso).
- Si `AGENT_RUNTIME_BACKEND=langchain` y `CHECKPOINT_BACKEND=postgres`, se activa checkpoint nativo de LangGraph (`AsyncPostgresSaver`) para memoria conversacional por `thread_id`.

## Pruebas con Docker (PostgreSQL real)

```bash
make docker-test-postgres
make docker-test-redis
make docker-down
```

## Flujo recomendado local (Bedrock + Chainlit)

```bash
# terminal 1
make run-api-bedrock

# terminal 2
make run-chainlit
```

## Flujo recomendado local (WhatsApp Twilio Sandbox)

```bash
# terminal 1
make run-api-bedrock

# terminal 2
ngrok http 8200
```

En Twilio Sandbox for WhatsApp:
- Configura `WHEN A MESSAGE COMES IN` con `https://<ngrok>/webhooks/whatsapp/twilio`
- Método: `HTTP POST`
- Envía un mensaje al número sandbox y valida la respuesta del agente.

## Flujo recomendado local (WhatsApp Meta Cloud API)

```bash
# terminal 1
make run-api-bedrock

# terminal 2
ngrok http 8200
```

En Meta App Dashboard (Webhooks):
- Callback URL: `https://<ngrok>/webhooks/whatsapp/meta`
- Verify token: debe coincidir con `WHATSAPP_META_VERIFY_TOKEN`
- Suscribe el campo `messages` del producto WhatsApp.
