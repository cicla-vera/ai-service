from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import getenv
from typing import Any

from app.schemas.analyze import (
    AnalysisProviderMetadata,
    AnalysisStatus,
    AnalyzeEvidenceRequest,
    AudioTranscription,
    TranscriptionSegment,
)
from app.services.audio_source import AudioSource

DEFAULT_TRANSCRIPTION_PROVIDER = "mock"
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"


@dataclass(frozen=True)
class TranscriptionResult:
    status: AnalysisStatus
    transcription: AudioTranscription | None
    provider_metadata: AnalysisProviderMetadata
    detected_signals: list[str]
    confidence: float
    failure_reason: str | None = None


class TranscriptionError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class BaseTranscriptionProvider:
    provider_name = "base"
    model_name = "unavailable"

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        raise NotImplementedError

    def metadata(self) -> AnalysisProviderMetadata:
        return AnalysisProviderMetadata(
            provider=self.provider_name,
            model=self.model_name,
            model_version=self.model_name,
        )


class MockTranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "mock"
    model_name = "mock-transcription"

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        if source is None:
            return TranscriptionResult(
                status=AnalysisStatus.COMPLETED,
                transcription=None,
                provider_metadata=self.metadata(),
                detected_signals=["transcription_skipped:no_audio_source"],
                confidence=0.12,
            )

        duration_ms = self._get_duration_ms(payload)
        transcription = AudioTranscription(
            text=f"Transcricao mock da evidencia {payload.evidence_record_id}.",
            language="pt",
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=duration_ms,
                    text=f"Transcricao mock da evidencia {payload.evidence_record_id}.",
                    confidence=0.99,
                )
            ],
        )

        return TranscriptionResult(
            status=AnalysisStatus.COMPLETED,
            transcription=transcription,
            provider_metadata=self.metadata(),
            detected_signals=[
                "transcription_completed",
                f"audio_source:{source.source_kind}",
            ],
            confidence=0.2,
        )

    def _get_duration_ms(self, payload: AnalyzeEvidenceRequest) -> int:
        context = payload.capture_context
        if not context or not context.capture_started_at or not context.capture_ended_at:
            return 0

        return max(
            0,
            round(
                (
                    context.capture_ended_at - context.capture_started_at
                ).total_seconds()
                * 1000,
            ),
        )


class OpenAITranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "openai"

    def __init__(self) -> None:
        self.model_name = getenv(
            "OPENAI_TRANSCRIPTION_MODEL",
            DEFAULT_OPENAI_TRANSCRIPTION_MODEL,
        )

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        if source is None:
            raise TranscriptionError("audio_source_missing")

        api_key = getenv("OPENAI_API_KEY")
        if not api_key:
            raise TranscriptionError("openai_api_key_missing")

        try:
            from openai import OpenAI, OpenAIError
        except ImportError as error:
            raise TranscriptionError("openai_sdk_missing") from error

        audio_file = BytesIO(source.data)
        audio_file.name = source.filename

        kwargs: dict[str, Any] = {
            "file": audio_file,
            "model": self.model_name,
            "response_format": "json",
        }
        language = getenv("OPENAI_TRANSCRIPTION_LANGUAGE")
        prompt = getenv("OPENAI_TRANSCRIPTION_PROMPT")

        if language:
            kwargs["language"] = language

        if prompt:
            kwargs["prompt"] = prompt

        try:
            response = OpenAI(api_key=api_key).audio.transcriptions.create(**kwargs)
        except OpenAIError as error:
            raise TranscriptionError("openai_transcription_failed") from error

        transcription = self._parse_response(response)
        text = transcription.text.strip()

        if not text:
            return TranscriptionResult(
                status=AnalysisStatus.INCONCLUSIVE,
                transcription=transcription,
                provider_metadata=self.metadata(),
                detected_signals=[
                    "transcription_completed",
                    "transcription_empty",
                    f"audio_source:{source.source_kind}",
                ],
                confidence=0.0,
            )

        return TranscriptionResult(
            status=AnalysisStatus.COMPLETED,
            transcription=transcription,
            provider_metadata=self.metadata(),
            detected_signals=[
                "transcription_completed",
                f"transcription_model:{self.model_name}",
                f"audio_source:{source.source_kind}",
            ],
            confidence=0.2,
        )

    def _parse_response(self, response: Any) -> AudioTranscription:
        text = self._get_value(response, "text")
        language = self._get_value(response, "language")
        segments = self._get_value(response, "segments") or []

        return AudioTranscription(
            text=text if isinstance(text, str) else "",
            language=language if isinstance(language, str) else None,
            segments=self._parse_segments(segments),
        )

    def _parse_segments(self, raw_segments: Any) -> list[TranscriptionSegment]:
        if not isinstance(raw_segments, list):
            return []

        segments: list[TranscriptionSegment] = []
        for item in raw_segments:
            text = self._get_value(item, "text")
            start = self._get_value(item, "start")
            end = self._get_value(item, "end")

            if not isinstance(text, str):
                continue

            segments.append(
                TranscriptionSegment(
                    start_ms=self._seconds_to_ms(start),
                    end_ms=self._seconds_to_ms(end),
                    text=text,
                    confidence=self._get_confidence(item),
                )
            )

        return segments

    def _get_value(self, target: Any, key: str) -> Any:
        if isinstance(target, dict):
            return target.get(key)

        return getattr(target, key, None)

    def _seconds_to_ms(self, value: Any) -> int:
        if isinstance(value, (int, float)) and value > 0:
            return round(value * 1000)

        return 0

    def _get_confidence(self, target: Any) -> float | None:
        confidence = self._get_value(target, "confidence")
        if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
            return float(confidence)

        return None


def get_transcription_provider() -> BaseTranscriptionProvider:
    provider = getenv("AI_TRANSCRIPTION_PROVIDER", DEFAULT_TRANSCRIPTION_PROVIDER)
    provider = provider.strip().lower()

    if provider == "openai":
        return OpenAITranscriptionProvider()

    return MockTranscriptionProvider()
