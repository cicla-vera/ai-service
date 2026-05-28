# Cicla Vera — AI Service

FastAPI microsservice used by the Vera safety layer to support evidence analysis.

## Requirements

- Python 3.12+
- uv

## Install

```bash
uv sync
cp .env.example .env
```

## Environment

The service defaults to deterministic mock transcription. Set these variables
when testing a real provider:

```env
AI_TRANSCRIPTION_PROVIDER=mock
AI_MOCK_TRANSCRIPTION_TEXT=
OPENAI_API_KEY=
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
OPENAI_TRANSCRIPTION_LANGUAGE=pt
OPENAI_TRANSCRIPTION_PROMPT=
DEEPGRAM_API_KEY=
DEEPGRAM_TRANSCRIPTION_MODEL=nova-2
DEEPGRAM_TRANSCRIPTION_LANGUAGE=pt-BR
DEEPGRAM_TRANSCRIPTION_BASE_URL=https://api.deepgram.com/v1/listen
DEEPGRAM_TRANSCRIPTION_TIMEOUT_SECONDS=30
AI_SERVICE_MAX_AUDIO_SOURCE_BYTES=26214400
AI_SERVICE_AUDIO_FETCH_TIMEOUT_SECONDS=10
AI_SERVICE_ALLOWED_AUDIO_HOSTS=
AI_SERVICE_ALLOW_INSECURE_AUDIO_REFERENCES=false
AI_SERVICE_ALLOW_FILE_REFERENCES=false
AI_ACOUSTIC_WINDOW_MS=100
AI_ACOUSTIC_HIGH_RMS_THRESHOLD=0.42
AI_ACOUSTIC_PEAK_THRESHOLD=0.82
AI_ACOUSTIC_IMPACT_DELTA_THRESHOLD=0.35
AI_ACOUSTIC_CLIPPING_RATIO_THRESHOLD=0.03
```

`AI_TRANSCRIPTION_PROVIDER=openai` uses OpenAI's audio transcription endpoint
through the official Python SDK. The default model is
`gpt-4o-mini-transcribe`; use `gpt-4o-transcribe` if accuracy is more important
than cost/latency for a specific environment.

`AI_TRANSCRIPTION_PROVIDER=deepgram` uses Deepgram's pre-recorded `/v1/listen`
endpoint. The default model is `nova-2` with `pt-BR` because that path has
documented Portuguese support; set `DEEPGRAM_TRANSCRIPTION_MODEL=nova-3` when
you want to evaluate the newer model for a specific environment.

`AI_MOCK_TRANSCRIPTION_TEXT` is optional and exists for local/dev tests that
need to simulate a specific transcript without spending provider credits.

The current `/analyze` JSON contract accepts `storageReference` in three forms:

- `data:audio/...;base64,...` for small local tests.
- `https://...` signed URLs from the backend/storage provider.
- `file://...` only when `AI_SERVICE_ALLOW_FILE_REFERENCES=true` for local dev.

Every resolved audio source is checked against `size` and the backend-provided
SHA-256 `contentHash` before transcription. Plain `http://` references are
blocked unless `AI_SERVICE_ALLOW_INSECURE_AUDIO_REFERENCES=true`; set
`AI_SERVICE_ALLOWED_AUDIO_HOSTS` to a comma-separated host allowlist in shared
or production-like environments.

The acoustic detector is intentionally conservative and deterministic in this
MVP phase. It currently inspects WAV/PCM clips for objective signals such as
sustained high amplitude, clipping, and sudden impact-like peaks. It does not
claim to identify emotion, prove aggression, or replace the later risk
aggregator; it only emits timestamped `acousticEvents` and compact
`detectedSignals` for the backend to audit and combine with other evidence.

The risk aggregator is also deterministic in this first MVP implementation. It
combines transcript phrase patterns, acoustic events, and capture context into
`LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` risk. `CRITICAL` is intentionally
reserved for concrete threat language, explicit emergency context, or a strong
combination of high-risk speech and impact-like acoustic signals. These outputs
are triage signals for Vera workflows, not legal conclusions or proof of a
crime.

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

The `/analyze` endpoint defines the v1 contract for Vera audio evidence
analysis. By default it uses deterministic mock transcription for integration
tests. When `AI_TRANSCRIPTION_PROVIDER=openai` or
`AI_TRANSCRIPTION_PROVIDER=deepgram` is configured with the corresponding API
key, it resolves `storageReference`, verifies hash/size, and sends the audio to
the configured speech-to-text model. It then runs a deterministic acoustic
detector and risk aggregator before returning the final classification.

Version `audio-evidence-v1` only accepts `AUDIO` evidence with an `audio/*`
MIME type. Later provider work can improve transcription, acoustic events,
threat matches, and risk classification from real model output without changing
the top-level response contract.

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
    "evidence_type:AUDIO",
    "transcription_skipped:no_audio_source",
    "acoustic_detection_skipped:no_audio_source",
    "risk_aggregation_completed",
    "risk_level:LOW"
  ],
  "shouldEscalate": false,
  "recommendedAction": "NONE",
  "evidenceWindow": {
    "startedAt": "2026-05-28T10:00:00Z",
    "endedAt": "2026-05-28T10:00:12Z",
    "durationMs": 12000
  },
  "transcription": null,
  "acousticEvents": [],
  "threatMatches": [],
  "providerMetadata": {
    "provider": "mock",
    "model": "mock-transcription",
    "modelVersion": "mock-transcription"
  },
  "processingStartedAt": "2026-05-28T10:00:13.000000Z",
  "processingFinishedAt": "2026-05-28T10:00:13.001000Z",
  "latencyMs": 1,
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
