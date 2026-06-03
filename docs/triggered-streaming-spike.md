# Triggered Audio Streaming Spike

Last reviewed: 2026-06-03.

## Decision

Do not stream ambient audio continuously in the MVP.

Keep the canonical MVP path as local trigger plus rolling pre-roll plus encrypted
evidence clip upload. Add near-real-time behavior by emitting short overlapping
clip chunks first. Treat provider streaming as a bounded acceleration path that
starts only after a local trigger and never replaces the legally preserved
evidence file.

Recommended order:

1. Implement hybrid chunked clips: 5-8 seconds of local pre-roll, then 10-15
   second chunks with 2 seconds of overlap while the event remains active.
2. Benchmark triggered provider streaming with consented fixtures only after
   chunked clips are stable on Android background capture.
3. Use streaming only for faster critical-alert detection. The full original
   clip remains the chain-of-custody artifact.

## Why

The product requirement is "do not miss the dangerous part", not "send every
second of ambient audio to a model". A phone can preserve a rolling buffer
locally at low cost, then upload only when the local sentinel sees voice,
volume, impact, scream, manual SOS, or other configured triggers.

Streaming can reduce time-to-transcript once triggered, but it introduces more
moving parts: provider session tokens, mobile WebSocket/WebRTC reliability,
network dropouts, data usage, battery, and privacy review. The evidence vault
still needs original audio, hash, upload acknowledgement, and metadata even if
streaming catches a threat phrase earlier.

## Compared Options

| Option | Latency | Cost | Evidence quality | Complexity | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Full continuous streaming | Best if network is perfect | Worst, billed while open | Needs separate full evidence path | Highest | Reject for MVP |
| Triggered provider streaming | Best after trigger | Bounded by session duration | Advisory unless paired with clip upload | High | Spike only |
| Short overlapping clip chunks | Good enough for MVP if chunks are 10-15s | Low | Strong, because original chunks are stored | Medium | Build next |
| Single completed clip upload | Slowest for escalation | Lowest | Strong | Low | Keep as fallback |

## Provider Notes

Prices and capabilities below are point-in-time planning data. Recheck provider
pricing before production or demos.

| Provider | Fit for triggered streaming | Price signal checked | Notes |
| --- | --- | --- | --- |
| AssemblyAI Universal-Streaming Multilingual | Strong MVP candidate | $0.15/hr; free tier advertises up to 333 streaming hours | Supports Portuguese among listed languages, temporary token auth, WebSocket streaming, turn results, and low price. Session billing depends on connection duration, so terminate aggressively. |
| Deepgram Nova-3 Multilingual | Strong MVP candidate | $0.0058/min streaming pay-as-you-go | Supports Portuguese/pt-BR, interim results, temporary JWT auth, high WSS concurrency, and useful endpointing/interim behavior. Good candidate when we need lower latency and strong Portuguese coverage. |
| OpenAI gpt-realtime-whisper | Best OpenAI realtime path, paid | $0.017/min | Designed for live transcript deltas and tunable latency. Use for demo/accuracy comparison if we can fund credits. Use server-minted ephemeral credentials or unified WebRTC; do not expose standard API keys in mobile. |
| Groq whisper-large-v3-turbo | Excellent clip fallback, not true streaming | $0.04/hr, minimum billed length 10s | Very cheap and fast for bounded audio files/chunks. Keep as a fallback for short clip uploads, not as the streaming candidate. |

Example cost for 100 triggered sessions/day at 90 seconds each:

| Provider/path | Daily audio/session time | Estimated daily cost |
| --- | --- | --- |
| AssemblyAI Universal-Streaming Multilingual | 150 minutes | $0.38 |
| Deepgram Nova-3 Multilingual streaming | 150 minutes | $0.87 |
| OpenAI gpt-realtime-whisper | 150 minutes | $2.55 |
| Groq chunked clips | 2.5 hours billed, ignoring minimum-request effects | $0.10 |

Groq can become more expensive than the rough number above if we send many
sub-10-second chunks because its speech-to-text endpoint has a 10-second
minimum billed length per request.

## Recommended MVP Flow

1. Mobile runs the local sentinel while Vera monitoring is active and consented.
2. Mobile keeps an encrypted rolling audio buffer locally, sized for 5-8 seconds
   of pre-roll.
3. On local trigger, mobile starts an evidence session and writes a canonical
   recording with pre-roll, trigger metadata, and location snapshots.
4. Mobile uploads chunks every 10-15 seconds with SHA-256, byte size, sequence
   number, previous chunk hash, capture window, and trigger reasons.
5. Backend stores every chunk as evidence and forwards the signed reference to
   `ai-service` for `/analyze`.
6. `ai-service` transcribes with the configured provider chain and returns
   `MEDIUM`, `HIGH`, or `CRITICAL` classifications per chunk.
7. Backend escalates emergency contacts only on `CRITICAL`, while continuing to
   collect the full clip until post-roll completes or the event is explicitly
   stopped.
8. If network drops, mobile keeps local encrypted evidence and retries upload.
   Streaming, if enabled, may fail without losing the canonical evidence.

## Triggered Streaming Prototype

Use this only after the chunked-clip path is stable.

Session start:

1. Local trigger fires.
2. Mobile creates/continues the canonical evidence clip.
3. Mobile asks backend for a provider session token.
4. Backend mints a short-lived provider token and binds it to the user/session.
5. Mobile opens provider WebSocket/WebRTC and streams audio for a bounded
   window.
6. Mobile forwards transcript deltas/finals to backend as advisory realtime
   events. Backend persists provider event IDs and timestamps.
7. `ai-service` evaluates transcript deltas/finals with the same risk policy as
   clip analysis.

Stop criteria:

- Stop after 60 seconds when risk remains `LOW` or `MEDIUM` and there is at
  least 8 seconds of local silence.
- Extend to 120 seconds when risk is `HIGH`.
- Extend to 180 seconds when risk is `CRITICAL` or contacts were escalated.
- Always terminate provider sessions explicitly.
- Always fall back to clip upload when token minting, stream connect,
  transcript events, or reconnect fail.

Pre-roll:

- Do not depend on streaming pre-roll for chain of custody.
- The canonical clip must include pre-roll.
- If the provider supports buffered audio safely, send pre-roll as tagged
  buffered chunks; otherwise send the live stream immediately and upload the
  pre-roll to `/analyze` as a normal clip chunk.

Security:

- Provider API keys stay server-side.
- Mobile receives only short-lived provider tokens.
- Provider tokens must be session-scoped where the provider supports it.
- Transcript events sent by mobile are advisory until matched to uploaded
  evidence chunks.
- Use hashed internal user/session identifiers in provider safety/user metadata
  when available.

## Measurement Plan

Evaluate on consented local fixtures and on controlled team recordings, not on
real victim audio.

Scenarios:

- benign quiet conversation
- TV/music/background noise
- tense argument without threat
- verbal abuse
- concrete threat
- distress call with impact-like noise
- network drop mid-event
- app backgrounded/locked during monitoring

Metrics:

- time from local trigger to first transcript text
- time from local trigger to final transcript for the first speech segment
- time from threat phrase end to `HIGH` or `CRITICAL`
- false positives and false negatives against the evaluation harness
- dropped audio duration during reconnection
- local evidence upload success rate
- provider session duration and estimated cost
- battery drain over a 30-minute armed monitoring run
- mobile data usage per 90-second triggered session

Suggested thresholds for the spike:

- first partial transcript within 2 seconds after live speech reaches provider
- first final transcript within 6 seconds for short turns
- critical decision within 5 seconds of a clear lethal threat in streaming mode
- critical decision within 20 seconds of a clear lethal threat in chunked mode
- zero lost canonical evidence when network is disabled during the event
- no provider API keys present in mobile logs, bundle, crash reports, or storage

## Platform And Privacy Constraints

Android:

- Microphone capture in the background needs an active foreground service and a
  visible notification.
- Android background-start restrictions are stricter for foreground services
  that require while-in-use permissions such as microphone and location. The
  app should start monitoring from an explicit user action while visible, then
  keep the foreground service alive.

iOS:

- Background recording requires the appropriate audio background mode and user
  microphone permission.
- iOS will still show system microphone indicators. The app must not attempt to
  hide platform privacy indicators.

Product/legal:

- The protected Vera layer should clearly state when monitoring and recording
  are active.
- Provider streaming should send only triggered windows, never continuous
  ambient audio.
- Evidence validity depends on backend storage, hash chain, timestamps,
  metadata, and encryption, not on provider transcript events alone.

## Follow-up Issues If Approved

Backend:

- Add endpoint to mint provider streaming tokens for a Vera evidence session.
- Persist streaming transcript events as advisory analysis records.
- Link streaming events to uploaded evidence chunk hashes.
- Add reconnect/session termination audit events.

Mobile:

- Add rolling encrypted pre-roll buffer.
- Emit sequenced 10-15 second evidence chunks with overlap.
- Add optional provider streaming client behind a remote config flag.
- Capture battery/data metrics during armed monitoring.
- Keep native notification and protected in-app disclosure aligned.

AI service:

- Add chunk-level analysis correlation by `alertSessionId` and chunk sequence.
- Add realtime transcript event analysis endpoint if backend forwards provider
  deltas.
- Extend the evaluation harness with latency fixtures for chunked vs streaming
  paths.

## Sources

- OpenAI Realtime overview: https://developers.openai.com/api/docs/guides/realtime
- OpenAI realtime transcription: https://developers.openai.com/api/docs/guides/realtime-transcription
- OpenAI WebRTC and ephemeral credentials: https://developers.openai.com/api/docs/guides/realtime-webrtc
- OpenAI Realtime costs: https://developers.openai.com/api/docs/guides/realtime-costs
- OpenAI API pricing: https://openai.com/api/pricing/
- Deepgram STT overview: https://developers.deepgram.com/docs/stt/getting-started
- Deepgram models/languages: https://developers.deepgram.com/docs/models-languages-overview
- Deepgram interim streaming results: https://developers.deepgram.com/docs/interim-results
- Deepgram token-based auth: https://developers.deepgram.com/guides/fundamentals/token-based-authentication
- Deepgram pricing: https://deepgram.com/pricing
- AssemblyAI streaming API: https://www.assemblyai.com/docs/api-reference/streaming-api/universal-streaming
- AssemblyAI universal streaming guide: https://www.assemblyai.com/docs/streaming/universal-streaming
- AssemblyAI pricing: https://www.assemblyai.com/pricing/
- Groq speech-to-text docs: https://console.groq.com/docs/speech-to-text
- Groq Whisper Large V3 Turbo model: https://console.groq.com/docs/model/whisper-large-v3-turbo
- Android foreground service restrictions: https://developer.android.com/about/versions/12/foreground-services
- Android foreground services overview: https://developer.android.com/develop/background-work/services/fgs
- Apple AVAudioSession recording category: https://developer.apple.com/documentation/avfaudio/avaudiosession/category-swift.struct/record
