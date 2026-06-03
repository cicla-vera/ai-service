from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from os import getenv
from time import monotonic, sleep
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
DEFAULT_TRANSCRIPTION_PROVIDER_CHAIN = ""
DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_MOCK_TRANSCRIPTION_TEXT = "Transcricao mock da evidencia {evidence_record_id}."
DEFAULT_DEEPGRAM_TRANSCRIPTION_MODEL = "nova-2"
DEFAULT_DEEPGRAM_TRANSCRIPTION_LANGUAGE = "pt-BR"
DEFAULT_DEEPGRAM_TRANSCRIPTION_BASE_URL = "https://api.deepgram.com/v1/listen"
DEFAULT_DEEPGRAM_TRANSCRIPTION_TIMEOUT_SECONDS = 30
DEFAULT_GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
DEFAULT_GROQ_TRANSCRIPTION_LANGUAGE = "pt"
DEFAULT_GROQ_TRANSCRIPTION_BASE_URL = (
    "https://api.groq.com/openai/v1/audio/transcriptions"
)
DEFAULT_GROQ_TRANSCRIPTION_TIMEOUT_SECONDS = 30
DEFAULT_ASSEMBLYAI_TRANSCRIPTION_BASE_URL = "https://api.assemblyai.com"
DEFAULT_ASSEMBLYAI_TRANSCRIPTION_SPEECH_MODELS = "universal-3-pro,universal-2"
DEFAULT_ASSEMBLYAI_TRANSCRIPTION_LANGUAGE_CODE = "pt"
DEFAULT_ASSEMBLYAI_TRANSCRIPTION_TIMEOUT_SECONDS = 60
DEFAULT_ASSEMBLYAI_TRANSCRIPTION_POLL_INTERVAL_SECONDS = 2

TERMINAL_TRANSCRIPTION_ERROR_SUFFIXES = (
    "_transcription_rejected",
    "_transcription_error",
)
TERMINAL_TRANSCRIPTION_ERROR_CODES = {
    "audio_source_missing",
    "unknown_transcription_provider",
}


@dataclass(frozen=True)
class TranscriptionResult:
    status: AnalysisStatus
    transcription: AudioTranscription | None
    provider_metadata: AnalysisProviderMetadata
    detected_signals: list[str]
    confidence: float
    failure_reason: str | None = None


class TranscriptionError(Exception):
    def __init__(
        self,
        code: str,
        *,
        fallback_allowed: bool | None = None,
        detected_signals: list[str] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.fallback_allowed = (
            self._get_default_fallback_allowed(code)
            if fallback_allowed is None
            else fallback_allowed
        )
        self.detected_signals = detected_signals or []

    def _get_default_fallback_allowed(self, code: str) -> bool:
        if code in TERMINAL_TRANSCRIPTION_ERROR_CODES:
            return False

        if code.endswith(TERMINAL_TRANSCRIPTION_ERROR_SUFFIXES):
            return False

        return True


class BaseTranscriptionProvider:
    provider_name = "base"
    model_name = "unavailable"

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        raise NotImplementedError

    def metadata(self, model_name: str | None = None) -> AnalysisProviderMetadata:
        model = model_name or self.model_name

        return AnalysisProviderMetadata(
            provider=self.provider_name,
            model=model,
            model_version=model,
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
            raise TranscriptionError(self._get_error_code(error)) from error

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

    def _get_error_code(self, error: Exception) -> str:
        status_code = getattr(error, "status_code", None)

        if status_code in {401, 403}:
            return "openai_auth_failed"

        if status_code == 429:
            return "openai_rate_limited"

        if isinstance(status_code, int) and 400 <= status_code < 500:
            return "openai_transcription_rejected"

        return "openai_transcription_failed"


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


class GroqTranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "groq"

    def __init__(self) -> None:
        self.model_name = getenv(
            "GROQ_TRANSCRIPTION_MODEL",
            DEFAULT_GROQ_TRANSCRIPTION_MODEL,
        )
        self.language = getenv(
            "GROQ_TRANSCRIPTION_LANGUAGE",
            DEFAULT_GROQ_TRANSCRIPTION_LANGUAGE,
        )
        self.base_url = getenv(
            "GROQ_TRANSCRIPTION_BASE_URL",
            DEFAULT_GROQ_TRANSCRIPTION_BASE_URL,
        )

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        if source is None:
            raise TranscriptionError("audio_source_missing")

        api_key = getenv("GROQ_API_KEY")
        if not api_key:
            raise TranscriptionError("groq_api_key_missing")

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
            ],
            confidence=self._get_confidence(transcription),
        )

    def _send_request(self, api_key: str, source: AudioSource) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {api_key}"}
        data: dict[str, Any] = {
            "model": self.model_name,
            "response_format": "verbose_json",
            "temperature": "0",
        }
        prompt = getenv("GROQ_TRANSCRIPTION_PROMPT")

        if self.language:
            data["language"] = self.language

        if prompt:
            data["prompt"] = prompt

        files = {
            "file": (
                source.filename,
                source.data,
                source.content_type,
            )
        }

        try:
            response = httpx.post(
                self.base_url,
                headers=headers,
                data=data,
                files=files,
                timeout=self._get_timeout_seconds(),
            )
        except httpx.HTTPError as error:
            raise TranscriptionError("groq_transcription_failed") from error

        if response.status_code in {401, 403}:
            raise TranscriptionError("groq_auth_failed")

        if response.status_code == 429:
            raise TranscriptionError("groq_rate_limited")

        if 400 <= response.status_code < 500:
            raise TranscriptionError("groq_transcription_rejected")

        if response.status_code >= 500:
            raise TranscriptionError("groq_transcription_failed")

        try:
            data = response.json()
        except ValueError as error:
            raise TranscriptionError("groq_invalid_response") from error

        if not isinstance(data, dict):
            raise TranscriptionError("groq_invalid_response")

        return data

    def _parse_response(self, response: dict[str, Any]) -> AudioTranscription:
        text = response.get("text")
        language = response.get("language")
        segments = response.get("segments")

        return AudioTranscription(
            text=text if isinstance(text, str) else "",
            language=language if isinstance(language, str) else self.language,
            segments=self._parse_segments(text, segments),
        )

    def _parse_segments(
        self,
        transcript: Any,
        raw_segments: Any,
    ) -> list[TranscriptionSegment]:
        if not isinstance(raw_segments, list) or not raw_segments:
            if not isinstance(transcript, str) or not transcript:
                return []

            return [
                TranscriptionSegment(
                    start_ms=0,
                    end_ms=0,
                    text=transcript,
                    confidence=None,
                )
            ]

        segments: list[TranscriptionSegment] = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if not isinstance(text, str):
                continue

            segments.append(
                TranscriptionSegment(
                    start_ms=self._seconds_to_ms(item.get("start")),
                    end_ms=self._seconds_to_ms(item.get("end")),
                    text=text,
                    confidence=self._get_segment_confidence(item),
                )
            )

        return segments

    def _get_confidence(self, transcription: AudioTranscription) -> float:
        confidences = [
            segment.confidence
            for segment in transcription.segments
            if segment.confidence is not None
        ]

        if not confidences:
            return 0.2

        return round(min(0.99, max(confidences)), 3)

    def _get_segment_confidence(self, item: dict[str, Any]) -> float | None:
        avg_logprob = item.get("avg_logprob")
        if not isinstance(avg_logprob, (int, float)):
            return None

        return round(max(0, min(0.99, 1 + float(avg_logprob))), 3)

    def _seconds_to_ms(self, value: Any) -> int:
        if isinstance(value, (int, float)) and value > 0:
            return round(value * 1000)

        return 0

    def _get_timeout_seconds(self) -> float:
        return _get_env_float(
            "GROQ_TRANSCRIPTION_TIMEOUT_SECONDS",
            DEFAULT_GROQ_TRANSCRIPTION_TIMEOUT_SECONDS,
        )


class AssemblyAITranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "assemblyai"

    def __init__(self) -> None:
        self.base_url = getenv(
            "ASSEMBLYAI_TRANSCRIPTION_BASE_URL",
            DEFAULT_ASSEMBLYAI_TRANSCRIPTION_BASE_URL,
        ).rstrip("/")
        self.speech_models = _get_env_list(
            "ASSEMBLYAI_TRANSCRIPTION_SPEECH_MODELS",
            DEFAULT_ASSEMBLYAI_TRANSCRIPTION_SPEECH_MODELS,
        )
        self.model_name = ",".join(self.speech_models)
        self.language_code = getenv(
            "ASSEMBLYAI_TRANSCRIPTION_LANGUAGE_CODE",
            DEFAULT_ASSEMBLYAI_TRANSCRIPTION_LANGUAGE_CODE,
        ).strip()

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        if source is None:
            raise TranscriptionError("audio_source_missing")

        api_key = getenv("ASSEMBLYAI_API_KEY")
        if not api_key:
            raise TranscriptionError("assemblyai_api_key_missing")

        upload_url = self._upload_audio(api_key=api_key, source=source)
        transcript_id = self._submit_transcript(
            api_key=api_key,
            audio_url=upload_url,
        )
        transcript = self._poll_transcript(
            api_key=api_key,
            transcript_id=transcript_id,
        )
        transcription = self._parse_response(transcript)
        model_name = self._get_model_name(transcript)

        if not transcription.text.strip():
            return TranscriptionResult(
                status=AnalysisStatus.INCONCLUSIVE,
                transcription=transcription,
                provider_metadata=self.metadata(model_name),
                detected_signals=[
                    "transcription_completed",
                    "transcription_empty",
                    f"audio_source:{source.source_kind}",
                    f"transcription_request_id:{transcript_id}",
                ],
                confidence=0,
            )

        return TranscriptionResult(
            status=AnalysisStatus.COMPLETED,
            transcription=transcription,
            provider_metadata=self.metadata(model_name),
            detected_signals=[
                "transcription_completed",
                f"transcription_model:{model_name}",
                f"audio_source:{source.source_kind}",
                f"transcription_request_id:{transcript_id}",
            ],
            confidence=self._get_confidence(transcription),
        )

    def _upload_audio(self, api_key: str, source: AudioSource) -> str:
        data = self._post_binary(
            path="/v2/upload",
            api_key=api_key,
            body=source.data,
        )
        upload_url = data.get("upload_url")

        if not isinstance(upload_url, str) or not upload_url:
            raise TranscriptionError("assemblyai_invalid_response")

        return upload_url

    def _submit_transcript(self, api_key: str, audio_url: str) -> str:
        body: dict[str, Any] = {
            "audio_url": audio_url,
            "speech_models": self.speech_models,
            "punctuate": True,
            "format_text": True,
        }

        if self.language_code:
            body["language_code"] = self.language_code
        else:
            body["language_detection"] = True

        data = self._post_json(
            path="/v2/transcript",
            api_key=api_key,
            body=body,
        )
        transcript_id = data.get("id")

        if not isinstance(transcript_id, str) or not transcript_id:
            raise TranscriptionError("assemblyai_invalid_response")

        return transcript_id

    def _poll_transcript(self, api_key: str, transcript_id: str) -> dict[str, Any]:
        timeout_seconds = _get_env_float(
            "ASSEMBLYAI_TRANSCRIPTION_TIMEOUT_SECONDS",
            DEFAULT_ASSEMBLYAI_TRANSCRIPTION_TIMEOUT_SECONDS,
        )
        poll_interval_seconds = _get_env_float(
            "ASSEMBLYAI_TRANSCRIPTION_POLL_INTERVAL_SECONDS",
            DEFAULT_ASSEMBLYAI_TRANSCRIPTION_POLL_INTERVAL_SECONDS,
        )
        deadline = monotonic() + timeout_seconds

        while True:
            data = self._get_json(
                path=f"/v2/transcript/{transcript_id}",
                api_key=api_key,
            )
            status = data.get("status")

            if status == "completed":
                return data

            if status == "error":
                raise TranscriptionError("assemblyai_transcription_error")

            remaining_seconds = deadline - monotonic()
            if remaining_seconds <= 0:
                raise TranscriptionError("assemblyai_polling_timeout")

            sleep(min(poll_interval_seconds, remaining_seconds))

    def _post_binary(
        self,
        *,
        path: str,
        api_key: str,
        body: bytes,
    ) -> dict[str, Any]:
        try:
            response = httpx.post(
                self._get_url(path),
                headers=self._get_headers(api_key),
                content=body,
                timeout=self._get_http_timeout_seconds(),
            )
        except httpx.HTTPError as error:
            raise TranscriptionError("assemblyai_transcription_failed") from error

        return self._parse_http_response(response)

    def _post_json(
        self,
        *,
        path: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            response = httpx.post(
                self._get_url(path),
                headers={
                    **self._get_headers(api_key),
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self._get_http_timeout_seconds(),
            )
        except httpx.HTTPError as error:
            raise TranscriptionError("assemblyai_transcription_failed") from error

        return self._parse_http_response(response)

    def _get_json(self, *, path: str, api_key: str) -> dict[str, Any]:
        try:
            response = httpx.get(
                self._get_url(path),
                headers=self._get_headers(api_key),
                timeout=self._get_http_timeout_seconds(),
            )
        except httpx.HTTPError as error:
            raise TranscriptionError("assemblyai_transcription_failed") from error

        return self._parse_http_response(response)

    def _parse_http_response(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code in {401, 403}:
            raise TranscriptionError("assemblyai_auth_failed")

        if response.status_code == 429:
            raise TranscriptionError("assemblyai_rate_limited")

        if 400 <= response.status_code < 500:
            raise TranscriptionError("assemblyai_transcription_rejected")

        if response.status_code >= 500:
            raise TranscriptionError("assemblyai_transcription_failed")

        try:
            data = response.json()
        except ValueError as error:
            raise TranscriptionError("assemblyai_invalid_response") from error

        if not isinstance(data, dict):
            raise TranscriptionError("assemblyai_invalid_response")

        return data

    def _parse_response(self, response: dict[str, Any]) -> AudioTranscription:
        text = response.get("text")
        language_code = response.get("language_code")
        words = response.get("words")

        return AudioTranscription(
            text=text if isinstance(text, str) else "",
            language=(
                language_code
                if isinstance(language_code, str) and language_code
                else self.language_code
            ),
            segments=self._parse_segments(text, words),
        )

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

        word_items = [word for word in words if isinstance(word, dict)]
        if not word_items:
            return []

        first_word = word_items[0]
        last_word = word_items[-1]

        return [
            TranscriptionSegment(
                start_ms=self._milliseconds_to_int(first_word.get("start")),
                end_ms=self._milliseconds_to_int(last_word.get("end")),
                text=transcript,
                confidence=self._get_words_confidence(word_items),
            )
        ]

    def _get_confidence(self, transcription: AudioTranscription) -> float:
        confidences = [
            segment.confidence
            for segment in transcription.segments
            if segment.confidence is not None
        ]

        if not confidences:
            return 0.2

        return round(min(0.99, max(confidences)), 3)

    def _get_words_confidence(self, words: list[dict[str, Any]]) -> float | None:
        confidences = [
            word.get("confidence")
            for word in words
            if isinstance(word.get("confidence"), (int, float))
            and 0 <= word.get("confidence") <= 1
        ]

        if not confidences:
            return None

        return round(sum(confidences) / len(confidences), 3)

    def _get_model_name(self, response: dict[str, Any]) -> str:
        speech_model_used = response.get("speech_model_used")

        if isinstance(speech_model_used, str) and speech_model_used:
            return speech_model_used

        return self.model_name

    def _milliseconds_to_int(self, value: Any) -> int:
        if isinstance(value, (int, float)) and value > 0:
            return round(value)

        return 0

    def _get_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": api_key}

    def _get_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get_http_timeout_seconds(self) -> float:
        return _get_env_float(
            "ASSEMBLYAI_HTTP_TIMEOUT_SECONDS",
            min(DEFAULT_ASSEMBLYAI_TRANSCRIPTION_TIMEOUT_SECONDS, 30),
        )


class FallbackTranscriptionProvider(BaseTranscriptionProvider):
    provider_name = "fallback"

    def __init__(self, provider_names: list[str]) -> None:
        self.provider_names = provider_names
        self.model_name = ">".join(provider_names)

    def transcribe(
        self,
        source: AudioSource | None,
        payload: AnalyzeEvidenceRequest,
    ) -> TranscriptionResult:
        detected_signals = [
            f"transcription_provider_chain:{'>'.join(self.provider_names)}",
        ]
        last_error: TranscriptionError | None = None

        for provider_name in self.provider_names:
            normalized_provider_name = provider_name.strip().lower()
            detected_signals.append(
                f"transcription_provider_attempt:{normalized_provider_name}",
            )

            try:
                provider = _create_transcription_provider(normalized_provider_name)
                result = provider.transcribe(source, payload)
            except TranscriptionError as error:
                last_error = error
                detected_signals.append(
                    "transcription_provider_failed:"
                    f"{normalized_provider_name}:{error.code}",
                )
                detected_signals.extend(error.detected_signals)

                if not error.fallback_allowed:
                    raise TranscriptionError(
                        error.code,
                        fallback_allowed=False,
                        detected_signals=detected_signals,
                    ) from error

                detected_signals.append(
                    f"transcription_fallback:{normalized_provider_name}:{error.code}",
                )
                continue

            return replace(
                result,
                detected_signals=[
                    *detected_signals,
                    f"transcription_provider_selected:{provider.provider_name}",
                    *result.detected_signals,
                ],
            )

        failure_reason = (
            last_error.code if last_error else "transcription_provider_chain_empty"
        )

        raise TranscriptionError(
            failure_reason,
            fallback_allowed=False,
            detected_signals=detected_signals,
        )


def get_transcription_provider() -> BaseTranscriptionProvider:
    provider_chain = _get_env_list(
        "AI_TRANSCRIPTION_PROVIDER_CHAIN",
        DEFAULT_TRANSCRIPTION_PROVIDER_CHAIN,
    )
    if provider_chain:
        return FallbackTranscriptionProvider(provider_chain)

    provider = getenv("AI_TRANSCRIPTION_PROVIDER", DEFAULT_TRANSCRIPTION_PROVIDER)
    provider = provider.strip().lower()

    return _create_transcription_provider(provider)


def _create_transcription_provider(provider: str) -> BaseTranscriptionProvider:
    if provider == "openai":
        return OpenAITranscriptionProvider()

    if provider == "deepgram":
        return DeepgramTranscriptionProvider()

    if provider == "groq":
        return GroqTranscriptionProvider()

    if provider == "assemblyai":
        return AssemblyAITranscriptionProvider()

    if provider == "mock":
        return MockTranscriptionProvider()

    raise TranscriptionError("unknown_transcription_provider")


def _get_env_list(name: str, default: str) -> list[str]:
    raw = getenv(name, default)
    if not raw:
        return []

    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _get_env_float(name: str, default: float) -> float:
    raw = getenv(name)
    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError:
        return default

    return value if value > 0 else default
