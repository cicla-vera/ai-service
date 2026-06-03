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
AI_TRANSCRIPTION_PROVIDER_CHAIN=
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
GROQ_API_KEY=
GROQ_TRANSCRIPTION_MODEL=whisper-large-v3-turbo
GROQ_TRANSCRIPTION_LANGUAGE=pt
GROQ_TRANSCRIPTION_PROMPT=
GROQ_TRANSCRIPTION_BASE_URL=https://api.groq.com/openai/v1/audio/transcriptions
GROQ_TRANSCRIPTION_TIMEOUT_SECONDS=30
ASSEMBLYAI_API_KEY=
ASSEMBLYAI_TRANSCRIPTION_BASE_URL=https://api.assemblyai.com
ASSEMBLYAI_TRANSCRIPTION_SPEECH_MODELS=universal-3-pro,universal-2
ASSEMBLYAI_TRANSCRIPTION_LANGUAGE_CODE=pt
ASSEMBLYAI_TRANSCRIPTION_TIMEOUT_SECONDS=60
ASSEMBLYAI_TRANSCRIPTION_POLL_INTERVAL_SECONDS=2
ASSEMBLYAI_HTTP_TIMEOUT_SECONDS=30
AI_SERVICE_MAX_AUDIO_SOURCE_BYTES=26214400
AI_SERVICE_AUDIO_FETCH_TIMEOUT_SECONDS=10
AI_SERVICE_ALLOWED_AUDIO_HOSTS=
AI_SERVICE_ALLOW_INSECURE_AUDIO_REFERENCES=false
AI_SERVICE_ALLOW_FILE_REFERENCES=false
AI_ACOUSTIC_WINDOW_MS=100
AI_ACOUSTIC_MAX_DURATION_SECONDS=120
AI_ACOUSTIC_HIGH_RMS_THRESHOLD=0.42
AI_ACOUSTIC_PEAK_THRESHOLD=0.82
AI_ACOUSTIC_IMPACT_DELTA_THRESHOLD=0.35
AI_ACOUSTIC_CLIPPING_RATIO_THRESHOLD=0.03
```

`AI_TRANSCRIPTION_PROVIDER_CHAIN` enables resilient provider fallback. When it
is set, it overrides `AI_TRANSCRIPTION_PROVIDER` and tries providers in order,
for example `deepgram,assemblyai,groq,openai`. Fallback is only used for
availability-style failures such as missing credentials, auth/provider outages,
rate limits, timeouts or invalid provider responses. Provider rejections for the
audio/request itself, such as unsupported or invalid media, stop the chain so a
bad evidence file is not hidden by another provider. The response records
attempted, failed, fallback and selected providers in `detectedSignals`.

`AI_TRANSCRIPTION_PROVIDER=openai` uses OpenAI's audio transcription endpoint
through the official Python SDK. The default model is
`gpt-4o-mini-transcribe`; use `gpt-4o-transcribe` if accuracy is more important
than cost/latency for a specific environment.

`AI_TRANSCRIPTION_PROVIDER=deepgram` uses Deepgram's pre-recorded `/v1/listen`
endpoint. The default model is `nova-2` with `pt-BR` because that path has
documented Portuguese support; set `DEEPGRAM_TRANSCRIPTION_MODEL=nova-3` when
you want to evaluate the newer model for a specific environment.

`AI_TRANSCRIPTION_PROVIDER=groq` uses Groq's OpenAI-compatible speech-to-text
endpoint. The default `whisper-large-v3-turbo` model is optimized for low
latency and low cost; use `whisper-large-v3` if accuracy matters more for a
specific test.

`AI_TRANSCRIPTION_PROVIDER=assemblyai` uploads the audio bytes to AssemblyAI,
submits an async transcript job and polls until completion or
`ASSEMBLYAI_TRANSCRIPTION_TIMEOUT_SECONDS`. The default speech model order is
`universal-3-pro,universal-2`, matching AssemblyAI's recommended fallback-style
model selection for broad language support.

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
MVP phase. It inspects WAV/PCM clips directly and normalizes the mobile
`.m4a`/AAC format to mono 16 kHz PCM in memory before detecting objective
signals such as sustained high amplitude, clipping, and sudden impact-like
peaks. The compressed original remains the evidence source and is not replaced
by this temporary analysis representation. Decoding stops after
`AI_ACOUSTIC_MAX_DURATION_SECONDS` to bound expansion of compressed input. The
detector does not claim to identify emotion, prove aggression, or replace the
later risk aggregator; it only emits timestamped `acousticEvents` and compact
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

## Evaluation harness

Use the evaluation harness before changing provider order, acoustic thresholds
or risk rules. The default mock mode runs public synthetic audio fixtures and
fixed transcripts, so it is safe for CI and does not spend provider credits:

```bash
uv run vera-ai-eval
uv run vera-ai-eval --format json --output evaluation/reports/mock.json
```

For a real provider or fallback chain, keep consented audio fixtures outside
git in `evaluation/local-fixtures/`, named by case id such as
`lethal_threat.m4a` or `distress_with_impact.wav`, then run:

```bash
AI_TRANSCRIPTION_PROVIDER_CHAIN=deepgram,assemblyai,groq,openai \
  uv run vera-ai-eval \
  --provider-mode configured \
  --fixtures-dir evaluation/local-fixtures \
  --output evaluation/reports/configured.md \
  --allow-failures
```

The report includes pass/fail, false positives, false negatives, current
thresholds, model/provider metadata, approximate latency, total audio seconds
and optional estimated cost. Set `AI_EVALUATION_COST_USD_PER_HOUR` or pass
`--cost-usd-per-hour` when you want cost estimates for a provider run.
Synthetic fixtures validate the current deterministic policy; real provider
mode is still required to evaluate transcription accuracy on consented test
audio.

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
tests. When a real provider is configured through `AI_TRANSCRIPTION_PROVIDER`
or a fallback order is configured through `AI_TRANSCRIPTION_PROVIDER_CHAIN`, it
resolves `storageReference`, verifies hash/size, and sends the audio to the
selected speech-to-text model. It then runs a deterministic acoustic detector
and risk aggregator before returning the final classification.

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
