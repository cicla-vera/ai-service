import base64
import wave
from hashlib import sha256
from io import BytesIO
from struct import pack

import av
import pytest
from fastapi.testclient import TestClient

from app.main import app


class FakeJsonResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> dict:
        return self.payload


@pytest.fixture(autouse=True)
def use_mock_transcription_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "mock")
    monkeypatch.delenv("AI_TRANSCRIPTION_PROVIDER_CHAIN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_MOCK_TRANSCRIPTION_TEXT", raising=False)


def test_analyze_returns_stable_mock_response() -> None:
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": 512,
            "contentHash": "a" * 64,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["analysisId"] == "mock-analysis-evidence-id"
    assert body["analysisVersion"] == "audio-evidence-v1"
    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "LOW"
    assert body["confidence"] == 0.12
    assert body["summary"] == (
        "Mock analysis completed. No real model was executed and no "
        "critical escalation was inferred from metadata-only input."
    )
    assert body["detectedSignals"] == [
        "mock_analysis",
        "metadata_received",
        "evidence_type:AUDIO",
        "transcription_skipped:no_audio_source",
        "acoustic_detection_skipped:no_audio_source",
        "risk_aggregation_completed",
        "risk_level:LOW",
    ]
    assert body["shouldEscalate"] is False
    assert body["recommendedAction"] == "NONE"
    assert body["evidenceWindow"] == {
        "startedAt": None,
        "endedAt": None,
        "durationMs": None,
    }
    assert body["transcription"] is None
    assert body["acousticEvents"] == []
    assert body["threatMatches"] == []
    assert body["providerMetadata"] == {
        "provider": "mock",
        "model": "mock-transcription",
        "modelVersion": "mock-transcription",
    }
    assert isinstance(body["processingStartedAt"], str)
    assert isinstance(body["processingFinishedAt"], str)
    assert isinstance(body["latencyMs"], int)
    assert body["failureReason"] is None


def test_analyze_validates_required_payload_fields() -> None:
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": 0,
            "contentHash": "short",
        },
    )

    assert response.status_code == 422
    errors = response.json()["detail"]
    error_fields = {tuple(error["loc"]) for error in errors}

    assert ("body", "evidenceRecordId") in error_fields
    assert ("body", "size") in error_fields
    assert ("body", "contentHash") in error_fields


def test_analyze_accepts_capture_context() -> None:
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/m4a",
            "size": 1024,
            "contentHash": "b" * 64,
            "captureContext": {
                "captureStartedAt": "2026-05-28T10:00:00Z",
                "captureEndedAt": "2026-05-28T10:00:12Z",
                "triggeredAt": "2026-05-28T10:00:03Z",
                "preRollMs": 5000,
                "postRollMs": 7000,
                "triggerReasons": ["voice_activity", "volume_spike"],
                "localConfidence": 0.77,
                "platform": "android",
                "foreground": False,
                "location": {
                    "latitude": -3.7319,
                    "longitude": -38.5267,
                    "accuracyMeters": 15,
                    "capturedAt": "2026-05-28T10:00:02Z",
                },
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["evidenceWindow"] == {
        "startedAt": "2026-05-28T10:00:00Z",
        "endedAt": "2026-05-28T10:00:12Z",
        "durationMs": 12000,
    }


def test_analyze_does_not_mark_volume_spike_context_as_high_risk() -> None:
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/m4a",
            "size": 1024,
            "contentHash": "b" * 64,
            "captureContext": {
                "triggerReasons": ["voice_activity", "volume_spike"],
                "localConfidence": 0.77,
                "platform": "android",
                "foreground": False,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["riskLevel"] == "MEDIUM"
    assert body["shouldEscalate"] is False
    assert body["recommendedAction"] == "STORE_EVIDENCE"
    assert "risk_level:MEDIUM" in body["detectedSignals"]
    assert "risk_context_trigger:volume_spike" in body["detectedSignals"]


def test_analyze_accepts_manual_transcription_text_for_demo_mode() -> None:
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "manual-evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/manual",
            "size": 1,
            "contentHash": "d" * 64,
            "manualTranscriptionText": "Eu vou te matar agora.",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["analysisId"] == "manual-analysis-manual-evidence-id"
    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "CRITICAL"
    assert body["shouldEscalate"] is True
    assert body["recommendedAction"] == "ESCALATE_CONTACTS"
    assert body["transcription"] == {
        "text": "Eu vou te matar agora.",
        "language": "pt",
        "segments": [
            {
                "startMs": 0,
                "endMs": 0,
                "text": "Eu vou te matar agora.",
                "confidence": 1,
            },
        ],
    }
    assert body["providerMetadata"] == {
        "provider": "manual",
        "model": "manual-transcription",
        "modelVersion": "manual-transcription-v1",
    }
    assert "transcription_source:manual" in body["detectedSignals"]
    assert "threat_signal:concrete_lethal_threat" in body["detectedSignals"]


def test_analyze_rejects_non_audio_contract_payload() -> None:
    client = TestClient(app)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "IMAGE",
            "mimeType": "image/jpeg",
            "size": 512,
            "contentHash": "c" * 64,
        },
    )

    assert response.status_code == 422
    assert "Audio analysis v1 only supports AUDIO evidence" in str(
        response.json()["detail"],
    )


def test_analyze_transcribes_mock_data_url_reference() -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "COMPLETED"
    assert body["summary"] == (
        "Audio transcription and acoustic analysis completed without threat "
        "signals in this first-pass classifier."
    )
    assert body["detectedSignals"] == [
        "metadata_received",
        "evidence_type:AUDIO",
        "transcription_completed",
        "audio_source:data_url",
        "acoustic_detection_skipped:invalid_wav",
        "risk_aggregation_completed",
        "risk_level:LOW",
    ]
    assert body["transcription"] == {
        "text": "Transcricao mock da evidencia evidence-id.",
        "language": "pt",
        "segments": [
            {
                "startMs": 0,
                "endMs": 0,
                "text": "Transcricao mock da evidencia evidence-id.",
                "confidence": 0.99,
            }
        ],
    }
    assert body["acousticEvents"] == []


def test_analyze_detects_relevant_acoustic_events_from_wav() -> None:
    client = TestClient(app)
    audio_bytes = _build_wav(
        [0] * 200
        + [32767] * 120
        + [0] * 100
        + [26000] * 300
        + [0] * 280,
    )
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()
    labels = {event["label"] for event in body["acousticEvents"]}

    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "HIGH"
    assert body["shouldEscalate"] is False
    assert body["recommendedAction"] == "REVIEW"
    assert body["summary"] == (
        "High-risk evidence candidate detected. Store the clip and request "
        "human review before any contact escalation."
    )
    assert "acoustic_detection_completed" in body["detectedSignals"]
    assert "acoustic_signal:relevant" in body["detectedSignals"]
    assert "acoustic_signal:critical_candidate" in body["detectedSignals"]
    assert "risk_aggregation_completed" in body["detectedSignals"]
    assert "risk_level:HIGH" in body["detectedSignals"]
    assert "risk_input:acoustic_critical_candidate" in body["detectedSignals"]
    assert "impact_candidate" in labels
    assert "clipping_detected" in labels
    assert "sustained_loud_audio" in labels


def test_analyze_detects_acoustic_events_from_mobile_m4a() -> None:
    client = TestClient(app)
    audio_bytes = _build_m4a(
        [0] * 1600
        + [32767] * 960
        + [0] * 800
        + [28000] * 2400
        + [0] * 2240,
    )
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/mp4",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/mp4;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()
    labels = {event["label"] for event in body["acousticEvents"]}

    assert body["status"] == "COMPLETED"
    assert "acoustic_normalization:mobile_audio_to_pcm_s16le" in body["detectedSignals"]
    assert "acoustic_normalization_sample_rate:16000" in body["detectedSignals"]
    assert "acoustic_detection_completed" in body["detectedSignals"]
    assert "acoustic_signal:relevant" in body["detectedSignals"]
    assert "sustained_loud_audio" in labels
    assert "impact_candidate" in labels


def test_analyze_skips_invalid_mobile_m4a_safely() -> None:
    client = TestClient(app)
    audio_bytes = b"invalid m4a bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/mp4",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/mp4;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "COMPLETED"
    assert body["acousticEvents"] == []
    assert "acoustic_detection_skipped:invalid_audio" in body["detectedSignals"]


def test_analyze_skips_unsupported_acoustic_format_safely() -> None:
    client = TestClient(app)
    audio_bytes = b"fake mp3 bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/mpeg",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/mpeg;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "COMPLETED"
    assert body["acousticEvents"] == []
    assert "acoustic_detection_skipped:unsupported_format" in body["detectedSignals"]


def test_analyze_stops_mobile_audio_normalization_after_duration_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = _build_m4a([12000] * 16_000)
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    monkeypatch.setenv("AI_ACOUSTIC_MAX_DURATION_SECONDS", "1")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/mp4",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/mp4;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "COMPLETED"
    assert body["acousticEvents"] == []
    assert "acoustic_detection_skipped:duration_limit_exceeded" in body[
        "detectedSignals"
    ]


def test_analyze_marks_verbal_abuse_as_relevant_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    monkeypatch.setenv("AI_MOCK_TRANSCRIPTION_TEXT", "Cala a boca sua vagabunda.")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "MEDIUM"
    assert body["shouldEscalate"] is False
    assert body["recommendedAction"] == "STORE_EVIDENCE"
    assert body["summary"] == (
        "Relevant evidence candidate detected. Store the clip with transcript "
        "and acoustic metadata for later review."
    )
    assert "risk_level:MEDIUM" in body["detectedSignals"]
    assert "threat_signal:severe_verbal_abuse" in body["detectedSignals"]
    assert body["threatMatches"][0]["label"] == "severe_verbal_abuse"
    assert body["threatMatches"][0]["severity"] == "MEDIUM"


def test_analyze_escalates_critical_concrete_threat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    monkeypatch.setenv("AI_MOCK_TRANSCRIPTION_TEXT", "Eu vou te matar agora.")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "CRITICAL"
    assert body["confidence"] == 0.95
    assert body["shouldEscalate"] is True
    assert body["recommendedAction"] == "ESCALATE_CONTACTS"
    assert body["summary"] == (
        "Critical risk candidate detected from concrete threat language. "
        "Escalation to emergency contacts is recommended."
    )
    assert "risk_level:CRITICAL" in body["detectedSignals"]
    assert "threat_signal:concrete_lethal_threat" in body["detectedSignals"]
    assert body["threatMatches"][0]["label"] == "concrete_lethal_threat"
    assert body["threatMatches"][0]["severity"] == "CRITICAL"
    assert body["threatMatches"][0]["evidence"] == "eu vou te matar"


def test_analyze_returns_failed_status_for_audio_hash_mismatch() -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": "d" * 64,
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "FAILED"
    assert body["riskLevel"] == "UNKNOWN"
    assert body["detectedSignals"] == ["analysis_failed", "content_hash_mismatch"]
    assert body["failureReason"] == "content_hash_mismatch"
    assert body["transcription"] is None


def test_analyze_returns_failed_status_when_openai_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "FAILED"
    assert body["failureReason"] == "openai_api_key_missing"


def test_analyze_returns_failed_status_when_deepgram_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "deepgram")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "FAILED"
    assert body["failureReason"] == "deepgram_api_key_missing"


def test_analyze_transcribes_deepgram_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    captured_request = {}
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test-key")

    class FakeDeepgramResponse:
        status_code = 200

        def json(self) -> dict:
            return {
                "metadata": {
                    "request_id": "dg-request-id",
                },
                "results": {
                    "channels": [
                        {
                            "detected_language": "pt-BR",
                            "alternatives": [
                                {
                                    "transcript": "Eu vou te matar agora.",
                                    "words": [
                                        {
                                            "word": "eu",
                                            "start": 0.0,
                                            "end": 0.1,
                                            "confidence": 0.94,
                                        },
                                        {
                                            "word": "agora",
                                            "start": 0.8,
                                            "end": 1.2,
                                            "confidence": 0.9,
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                },
            }

    def fake_post(url: str, **kwargs):
        captured_request["url"] = url
        captured_request.update(kwargs)
        return FakeDeepgramResponse()

    monkeypatch.setattr("app.services.transcription_service.httpx.post", fake_post)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert "model=nova-2" in captured_request["url"]
    assert "language=pt-BR" in captured_request["url"]
    assert captured_request["headers"]["Authorization"] == "Token dg-test-key"
    assert captured_request["headers"]["Content-Type"] == "audio/wav"
    assert captured_request["content"] == audio_bytes
    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "CRITICAL"
    assert body["shouldEscalate"] is True
    assert body["providerMetadata"] == {
        "provider": "deepgram",
        "model": "nova-2",
        "modelVersion": "nova-2",
    }
    assert body["transcription"] == {
        "text": "Eu vou te matar agora.",
        "language": "pt-BR",
        "segments": [
            {
                "startMs": 0,
                "endMs": 1200,
                "text": "Eu vou te matar agora.",
                "confidence": 0.92,
            }
        ],
    }
    assert "transcription_model:nova-2" in body["detectedSignals"]
    assert "transcription_request_id:dg-request-id" in body["detectedSignals"]
    assert "threat_signal:concrete_lethal_threat" in body["detectedSignals"]


def test_analyze_falls_back_from_deepgram_to_groq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    captured_request = {}
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER_CHAIN", "deepgram,groq")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

    def fake_post(url: str, **kwargs):
        captured_request["url"] = url
        captured_request.update(kwargs)
        return FakeJsonResponse(
            200,
            {
                "text": "Eu vou te matar agora.",
                "language": "pt",
                "segments": [
                    {
                        "start": 0,
                        "end": 1.2,
                        "text": "Eu vou te matar agora.",
                        "avg_logprob": -0.08,
                    }
                ],
            },
        )

    monkeypatch.setattr("app.services.transcription_service.httpx.post", fake_post)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert captured_request["url"] == (
        "https://api.groq.com/openai/v1/audio/transcriptions"
    )
    assert captured_request["headers"]["Authorization"] == "Bearer groq-test-key"
    assert captured_request["data"]["model"] == "whisper-large-v3-turbo"
    assert captured_request["data"]["language"] == "pt"
    assert captured_request["files"]["file"] == (
        "evidence-id.audio",
        audio_bytes,
        "audio/wav",
    )
    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "CRITICAL"
    assert body["providerMetadata"] == {
        "provider": "groq",
        "model": "whisper-large-v3-turbo",
        "modelVersion": "whisper-large-v3-turbo",
    }
    assert "transcription_provider_chain:deepgram>groq" in body["detectedSignals"]
    assert (
        "transcription_provider_failed:deepgram:deepgram_api_key_missing"
        in body["detectedSignals"]
    )
    assert (
        "transcription_fallback:deepgram:deepgram_api_key_missing"
        in body["detectedSignals"]
    )
    assert "transcription_provider_selected:groq" in body["detectedSignals"]


def test_analyze_does_not_fallback_after_terminal_provider_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    requested_urls: list[str] = []
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER_CHAIN", "deepgram,groq")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

    def fake_post(url: str, **kwargs):
        requested_urls.append(url)
        return FakeJsonResponse(400, {"error": "unsupported media"})

    monkeypatch.setattr("app.services.transcription_service.httpx.post", fake_post)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert requested_urls == [
        "https://api.deepgram.com/v1/listen?model=nova-2&language=pt-BR&"
        "smart_format=true&punctuate=true",
    ]
    assert body["status"] == "FAILED"
    assert body["failureReason"] == "deepgram_transcription_rejected"
    assert (
        "transcription_provider_failed:deepgram:deepgram_transcription_rejected"
        in body["detectedSignals"]
    )
    assert not any(
        "transcription_provider_selected:groq" == signal
        for signal in body["detectedSignals"]
    )


def test_analyze_transcribes_assemblyai_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    audio_bytes = b"fake audio bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    requests: list[tuple[str, dict]] = []
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "assemblyai")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "assembly-test-key")

    def fake_post(url: str, **kwargs):
        requests.append((url, kwargs))
        if url.endswith("/v2/upload"):
            return FakeJsonResponse(200, {"upload_url": "https://upload.example/audio"})

        if url.endswith("/v2/transcript"):
            return FakeJsonResponse(200, {"id": "assembly-transcript-id"})

        raise AssertionError(f"Unexpected POST {url}")

    def fake_get(url: str, **kwargs):
        requests.append((url, kwargs))
        return FakeJsonResponse(
            200,
            {
                "id": "assembly-transcript-id",
                "status": "completed",
                "text": "Eu vou te matar agora.",
                "language_code": "pt",
                "speech_model_used": "universal-3-pro",
                "words": [
                    {
                        "text": "Eu",
                        "start": 0,
                        "end": 100,
                        "confidence": 0.96,
                    },
                    {
                        "text": "agora",
                        "start": 900,
                        "end": 1200,
                        "confidence": 0.94,
                    },
                ],
            },
        )

    monkeypatch.setattr("app.services.transcription_service.httpx.post", fake_post)
    monkeypatch.setattr("app.services.transcription_service.httpx.get", fake_get)

    response = client.post(
        "/analyze",
        json={
            "evidenceRecordId": "evidence-id",
            "alertSessionId": "session-id",
            "evidenceType": "AUDIO",
            "mimeType": "audio/wav",
            "size": len(audio_bytes),
            "contentHash": sha256(audio_bytes).hexdigest(),
            "storageReference": f"data:audio/wav;base64,{encoded_audio}",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert requests[0][0] == "https://api.assemblyai.com/v2/upload"
    assert requests[0][1]["headers"]["Authorization"] == "assembly-test-key"
    assert requests[0][1]["content"] == audio_bytes
    assert requests[1][0] == "https://api.assemblyai.com/v2/transcript"
    assert requests[1][1]["json"] == {
        "audio_url": "https://upload.example/audio",
        "speech_models": ["universal-3-pro", "universal-2"],
        "punctuate": True,
        "format_text": True,
        "language_code": "pt",
    }
    assert requests[2][0] == (
        "https://api.assemblyai.com/v2/transcript/assembly-transcript-id"
    )
    assert body["status"] == "COMPLETED"
    assert body["riskLevel"] == "CRITICAL"
    assert body["providerMetadata"] == {
        "provider": "assemblyai",
        "model": "universal-3-pro",
        "modelVersion": "universal-3-pro",
    }
    assert body["transcription"] == {
        "text": "Eu vou te matar agora.",
        "language": "pt",
        "segments": [
            {
                "startMs": 0,
                "endMs": 1200,
                "text": "Eu vou te matar agora.",
                "confidence": 0.95,
            }
        ],
    }
    assert "transcription_model:universal-3-pro" in body["detectedSignals"]
    assert (
        "transcription_request_id:assembly-transcript-id"
        in body["detectedSignals"]
    )


def _build_wav(samples: list[int], frame_rate: int = 1000) -> bytes:
    buffer = BytesIO()

    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(frame_rate)
        audio.writeframes(b"".join(pack("<h", sample) for sample in samples))

    return buffer.getvalue()


def _build_m4a(samples: list[int], frame_rate: int = 8000) -> bytes:
    buffer = BytesIO()

    with av.open(buffer, mode="w", format="mp4") as output:
        stream = output.add_stream("aac", rate=frame_rate)
        stream.layout = "mono"
        frame = av.AudioFrame(format="s16", layout="mono", samples=len(samples))
        frame.sample_rate = frame_rate
        frame.planes[0].update(b"".join(pack("<h", sample) for sample in samples))

        for packet in stream.encode(frame):
            output.mux(packet)

        for packet in stream.encode():
            output.mux(packet)

    return buffer.getvalue()
