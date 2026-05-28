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

## Audio evidence analysis contract

The first `/analyze` endpoint defines the v1 contract for Vera audio evidence
analysis. The implementation is still intentionally mocked: it validates the
request shape used by the backend and returns a stable response for integration
tests. It does not run a real model and never infers critical escalation.

Version `audio-evidence-v1` only accepts `AUDIO` evidence with an `audio/*`
MIME type. Later provider work will fill transcription, acoustic events, threat
matches, and risk classification from real model output without changing the
top-level response contract.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "evidenceRecordId": "evidence-id",
    "alertSessionId": "session-id",
    "evidenceType": "AUDIO",
    "mimeType": "audio/wav",
    "size": 512,
    "contentHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "storageReference": "backend-owned-reference",
    "captureContext": {
      "captureStartedAt": "2026-05-28T10:00:00Z",
      "captureEndedAt": "2026-05-28T10:00:12Z",
      "triggeredAt": "2026-05-28T10:00:03Z",
      "preRollMs": 5000,
      "postRollMs": 7000,
      "triggerReasons": ["voice_activity", "volume_spike"],
      "localConfidence": 0.77,
      "platform": "android",
      "foreground": false,
      "location": {
        "latitude": -3.7319,
        "longitude": -38.5267,
        "accuracyMeters": 15,
        "capturedAt": "2026-05-28T10:00:02Z"
      }
    }
  }'
```

Expected response:

```json
{
  "analysisId": "mock-analysis-evidence-id",
  "analysisVersion": "audio-evidence-v1",
  "status": "COMPLETED",
  "riskLevel": "LOW",
  "confidence": 0.12,
  "summary": "Mock analysis completed. No real model was executed and no critical escalation was inferred from metadata-only input.",
  "detectedSignals": [
    "mock_analysis",
    "metadata_received",
    "evidence_type:AUDIO"
  ],
  "shouldEscalate": false,
  "recommendedAction": "NONE",
  "evidenceWindow": {
    "startedAt": "2026-05-28T10:00:00Z",
    "endedAt": "2026-05-28T10:00:12Z",
    "durationMs": 12000
  },
  "transcription": null,
  "acousticEvents": [
    {
      "label": "mock_metadata_only_analysis",
      "startMs": 0,
      "endMs": 0,
      "confidence": 0.12,
      "source": "mock"
    }
  ],
  "threatMatches": [],
  "providerMetadata": {
    "provider": "mock",
    "model": "metadata-only",
    "modelVersion": "audio-evidence-v1"
  },
  "processingStartedAt": "2026-01-01T00:00:00Z",
  "processingFinishedAt": "2026-01-01T00:00:01Z",
  "latencyMs": 1000,
  "failureReason": null
}
```

Top-level statuses are `QUEUED`, `PROCESSING`, `COMPLETED`, `FAILED`, and
`INCONCLUSIVE`. Risk levels are `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, and
`UNKNOWN`. `detectedSignals` intentionally remains a string array so the NestJS
backend can consume a stable compact summary, while richer details live in
`transcription`, `acousticEvents`, and `threatMatches`.

## Tests

```bash
uv run pytest
```
