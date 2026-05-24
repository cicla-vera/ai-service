# Cicla Vera — AI Service

FastAPI microsservice used by the Vera safety layer to support evidence analysis.

## Requirements

- Python 3.12+
- uv

## Install

```bash
uv sync
```

## Run locally

```bash
uv run ai-service
```

The service starts at `http://localhost:8000`.

## Health check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "ai-service"
}
```

## Tests

```bash
uv run pytest
```
