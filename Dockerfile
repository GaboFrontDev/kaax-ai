FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY app ./app
COPY migrations ./migrations

RUN uv pip install --system '.[bedrock,postgres,redis,migrations]'

EXPOSE 8200

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8200"]
