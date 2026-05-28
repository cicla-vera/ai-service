import base64
import wave
from hashlib import sha256
from io import BytesIO
from struct import pack

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def use_mock_transcription_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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


def _build_wav(samples: list[int], frame_rate: int = 1000) -> bytes:
    buffer = BytesIO()

    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(frame_rate)
        audio.writeframes(b"".join(pack("<h", sample) for sample in samples))

    return buffer.getvalue()
