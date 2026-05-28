from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from os import getenv
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

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
DEFAULT_MOCK_TRANSCRIPTION_TEXT = "Transcricao mock da evidencia {evidence_record_id}."
DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL = "nova-2"
DEFAULT_DEEPGRAM_TRANSCRIPTION_LANGUAGE = "pt-BR"
DEFAULT_DEEPGRAM_TRANSCRIPTION_BASE_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_DEEPGRAM_TRANSCRIPTION_TIMEOUT_SECONDS = 30


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
        text = getenv(
            "AI_MOCK_TRANSCRIPTION_TEXT",
        ) or DEFAULT_MOCK_TRANSCRIPTION_TEXT.format(
            evidence_record_id=payload.evidence_record_id,
        )

        transcription = AudioTranscription(
            text=text,
            language="pt",
            segments=[
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=duration_ms,
                    text=text,
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


class DeepgramTranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "deepgram"

    def __init__(self) -> None:
        self.model_name = getenv(
            "DEEPGRAM_TRANSCRIPTION_MODEL",
            DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL,
        )
        self.language = getenv(
            "DEEPGRAM_TRANSCRIPTION_LANGUAGE",
            DEFAULT_DEEPGRAM_TRANSCRIPTION_LANGUAGE,
        )
        self.base_url = getenv(
            "DEEPGRAM_TRANSCRIPTION_BASE_URL",
            DEFAULT_DEEPGRAM_TRANSCRIPTION_BASE_URL,
        )

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        if source is None:
            raise TranscriptionError("audio_source_missing")

        api_key = getenv("DEEPGRAM_API_KEY")
        if not api_key:
            raise TranscriptionError("deepgram_api_key_missing")

        response = self._send_request(api_key=api_key, source=source)
        transcription = self._parse_response(response)

        if not transcription.text.strip():
            return TranscriptionResult(
                status=AnalysisStatus.INCONCLUSIVE,
                transcription=transcription,
                provider_metadata=self.metadata(),
                detected_signals=[
                    "transcription_completed",
                    "transcription_empty",
                    f"audio_source:{source.source_kind}",
                ],
                confidence=0,
            )

        return TranscriptionResult(
            status=AnalysisStatus.COMPLETED,
            transcription=transcription,
            provider_metadata=self.metadata(),
            detected_signals=[
                "transcription_completed",
                f"transcription_model:{self.model_name}",
                f"audio_source:{source.source_kind}",
                *self._get_request_signal(response),
            ],
            confidence=self._get_confidence(transcription),
        )

    def _send_request(self, api_key: str, source: AudioSource) -> dict[str, Any]:
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": source.content_type,
        }
        params = {
            "model": self.model_name,
            "language": self.language,
            "smart_format": "true",
            "punctuate": "true",
        }

        try:
            response = httpx.post(
                self._get_url(params),
                headers=headers,
                content=source.data,
                timeout=self._get_timeout_seconds(),
            )
        except httpx.HTTPError as error:
            raise TranscriptionError("deepgram_transcription_failed") from error

        if response.status_code in {401, 403}:
            raise TranscriptionError("deepgram_auth_failed")

        if response.status_code == 429:
            raise TranscriptionError("deepgram_rate_limited")

        if 400 <= response.status_code < 500:
            raise TranscriptionError("deepgram_transcription_rejected")

        if response.status_code >= 500:
            raise TranscriptionError("deepgram_transcription_failed")

        try:
            data = response.json()
        except ValueError as error:
            raise TranscriptionError("deepgram_invalid_response") from error

        if not isinstance(data, dict):
            raise TranscriptionError("deepgram_invalid_response")

        return data

    def _parse_response(self, response: dict[str, Any]) -> AudioTranscription:
        alternative = self._get_primary_alternative(response)
        transcript = alternative.get("transcript")
        words = alternative.get("words")

        return AudioTranscription(
            text=transcript if isinstance(transcript, str) else "",
            language=self._get_language(response),
            segments=self._parse_segments(transcript, words),
        )

    def _get_primary_alternative(self, response: dict[str, Any]) -> dict[str, Any]:
        results = response.get("results")
        if not isinstance(results, dict):
            return {}

        channels = results.get("channels")
        if not isinstance(channels, list) or not channels:
            return {}

        first_channel = channels[0]
        if not isinstance(first_channel, dict):
            return {}

        alternatives = first_channel.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            return {}

        first_alternative = alternatives[0]

        return first_alternative if isinstance(first_alternative, dict) else {}

    def _parse_segments(
        self,
        transcript: Any,
        words: Any,
    ) -> list[TranscriptionSegment]:
        if not isinstance(transcript, str) or not transcript:
            return []

        if not isinstance(words, list) or not words:
            return [
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=0,
                    text=transcript,
                    confidence=None,
                )
            ]

        first_word = words[0] if isinstance(words[0], dict) else {}
        last_word = words[-1] if isinstance(words[-1], dict) else {}

        return [
            TranscriptionSegment(
                start_ms=self._seconds_to_ms(first_word.get("start")),
                end_ms=self._seconds_to_ms(last_word.get("end")),
                text=transcript,
                confidence=self._get_words_confidence(words),
            )
        ]

    def _get_language(self, response: dict[str, Any]) -> str:
        results = response.get("results")
        if isinstance(results, dict):
            channels = results.get("channels")
            if isinstance(channels, list) and channels:
                channel = channels[0]
                if isinstance(channel, dict):
                    detected_language = channel.get("detected_language")
                    if isinstance(detected_language, str) and detected_language:
                        return detected_language

        return self.language

    def _get_request_signal(self, response: dict[str, Any]) -> list[str]:
        metadata = response.get("metadata")
        if not isinstance(metadata, dict):
            return []

        request_id = metadata.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return []

        return [f"transcription_request_id:{request_id}"]

    def _get_confidence(self, transcription: AudioTranscription) -> float:
        confidences = [
            segment.confidence
            for segment in transcription.segments
            if segment.confidence is not None
        ]

        if not confidences:
            return 0.2

        return round(min(0.99, max(confidences)), 3)

    def _get_words_confidence(self, words: list[Any]) -> float | None:
        confidences = [
            word.get("confidence")
            for word in words
            if isinstance(word, dict)
            and isinstance(word.get("confidence"), (int, float))
            and 0 <= word.get("confidence") <= 1
        ]

        if not confidences:
            return None

        return round(sum(confidences) / len(confidences), 3)

    def _seconds_to_ms(self, value: Any) -> int:
        if isinstance(value, (int, float)) and value > 0:
            return round(value * 1000)

        return 0

    def _get_timeout_seconds(self) -> float:
        raw = getenv("DEEPGRAM_TRANSCRIPTION_TIMEOUT_SECONDS")
        if not raw:
            return DEFAULT_DEEPGRAM_TRANSCRIPTION_TIMEOUT_SECONDS

        try:
            value = float(raw)
        except ValueError:
            return DEFAULT_DEEPGRAM_TRANSCRIPTION_TIMEOUT_SECONDS

        return value if value > 0 else DEFAULT_DEEPGRAM_TRANSCRIPTION_TIMEOUT_SECONDS

    def _get_url(self, params: dict[str, str]) -> str:
        parts = urlsplit(self.base_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.update(params)

        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(query),
                parts.fragment,
            )
        )


def get_transcription_provider() -> BaseTranscriptionProvider:
    provider = getenv("AI_TRANSCRIPTION_PROVIDER", DEFAULT_TRANSCRIPTION_PROVIDER)
    provider = provider.strip().lower()

    if provider == "openai":
        return OpenAITranscriptionProvider()

    if provider == "deepgram":
        return DeepgramTranscriptionProvider()

    return MockTranscriptionProvider()
