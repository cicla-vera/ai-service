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

## Evidence analysis contract

The first `/analyze` endpoint is intentionally mocked. It validates the request
shape used by the backend and returns a stable response for integration tests.
It does not run a real model and never infers critical escalation.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "evidenceRecordId": "evidence-id",
    "alertSessionId": "session-id",
    "evidenceType": "AUDIO",
    "mimeType": "audio/wav",
    "size": 512,
    "contentHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  }'
```

Expected response:

```json
{
  "riskLevel": "LOW",
  "confidence": 0.12,
  "summary": "Mock analysis completed. No real model was executed and no critical escalation was inferred from metadata-only input.",
  "detectedSignals": [
    "mock_analysis",
    "metadata_received",
    "evidence_type:AUDIO"
  ],
  "shouldEscalate": false
}
```

## Tests

```bash
uv run pytest
```
