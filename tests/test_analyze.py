import base64
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def use_mock_transcription_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


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
    ]
    assert body["shouldEscalate"] is False
    assert body["recommendedAction"] == "NONE"
    assert body["evidenceWindow"] == {
        "startedAt": None,
        "endedAt": None,
        "durationMs": None,
    }
    assert body["transcription"] is None
    assert body["acousticEvents"] == [
        {
            "label": "mock_metadata_only_analysis",
            "startMs": 0,
            "endMs": 0,
            "confidence": 0.12,
            "source": "mock",
        }
    ]
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
        "Audio transcription completed. No threat or acoustic risk "
        "classification has run yet."
    )
    assert body["detectedSignals"] == [
        "metadata_received",
        "evidence_type:AUDIO",
        "transcription_completed",
        "audio_source:data_url",
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
