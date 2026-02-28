PYTHON ?= python3

.PHONY: run-api run-api-bedrock run-chainlit unit-tests integration-tests test \
	db-upgrade db-downgrade db-current \
	docker-up docker-down docker-logs docker-test-postgres \
	docker-up-redis docker-test-redis \
	cdk-bootstrap cdk-deploy cdk-diff cdk-destroy cdk-sync-secrets awsctl

ENV ?= dev
AGENT ?= default
AWSCTL_ARGS ?= help

run-api:
	source ./.env.local && uv run uvicorn app.api.main:app --host 0.0.0.0 --port 8200 --reload

run-api-bedrock:
	source ./.env.local && uv run --extra dev --extra bedrock --extra postgres uvicorn app.api.main:app --host 0.0.0.0 --port 8200 --reload

run-chainlit:
	source ./.env.local && uv run --extra dev --extra chainlit chainlit run app/channels/chainlit/app.py -w --port 8300

unit-tests:
	uv run --extra dev pytest app/tests/unit

integration-tests:
	uv run --extra dev pytest app/tests/integration

test:
	uv run --extra dev pytest

db-upgrade:
	source ./.env.local && uv run --extra migrations alembic upgrade head

db-downgrade:
	source ./.env.local && uv run --extra migrations alembic downgrade -1

db-current:
	source ./.env.local && uv run --extra migrations alembic current

docker-up:
	docker compose up -d postgres

docker-up-redis:
	docker compose up -d redis-master redis-sentinel-1 redis-sentinel-2 redis-sentinel-3

docker-down:
	docker compose down -v

docker-logs:
	docker compose logs --tail=200

docker-test-postgres: docker-up
	@echo "Waiting for postgres healthcheck..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' kaaxai-postgres 2>/dev/null)" = "healthy" ]; do sleep 1; done
	CHECKPOINT_BACKEND=postgres \
	DB_HOST=127.0.0.1 \
	DB_PORT=55432 \
	DB_USER=postgres \
	DB_PASSWORD=postgres \
	DB_NAME=postgres \
	uv run --extra dev --extra postgres pytest app/tests/integration/test_postgres_backend.py

docker-test-redis: docker-up-redis
	@echo "Waiting for redis sentinels healthcheck..."
	@for svc in kaaxai-redis-sentinel-1 kaaxai-redis-sentinel-2 kaaxai-redis-sentinel-3; do \
		ok=0; \
		for i in $$(seq 1 60); do \
			status=$$(docker inspect -f '{{.State.Health.Status}}' $$svc 2>/dev/null || echo "missing"); \
			if [ "$$status" = "healthy" ]; then \
				ok=1; \
				break; \
			fi; \
			sleep 1; \
		done; \
		if [ "$$ok" -ne 1 ]; then \
			echo "$$svc did not become healthy in time"; \
			docker compose logs --tail=200 redis-master redis-sentinel-1 redis-sentinel-2 redis-sentinel-3; \
			exit 1; \
		fi; \
	done
	ATTACHMENT_BACKEND=redis \
	MESSAGE_QUEUE_BACKEND=redis \
	REDIS_MASTER_NAME=mymaster \
	REDIS_SENTINELS=127.0.0.1:56379,127.0.0.1:56380,127.0.0.1:56381 \
	REDIS_MASTER_HOST_OVERRIDE=127.0.0.1 \
	REDIS_MASTER_PORT_OVERRIDE=56378 \
	uv run --extra dev --extra redis pytest app/tests/integration/test_redis_backend.py

cdk-bootstrap:
	./ops/bootstrap.sh

cdk-deploy:
	./ops/deploy.sh $(ENV) $(AGENT)

cdk-diff:
	./ops/diff.sh $(ENV) $(AGENT)

cdk-destroy:
	./ops/destroy.sh $(ENV) $(AGENT)

cdk-sync-secrets:
	./ops/secrets-sync.sh $(CDK_SECRET_NAME)

awsctl:
	./ops/awsctl.sh $(AWSCTL_ARGS)
