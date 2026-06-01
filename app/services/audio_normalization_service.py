from __future__ import annotations

import wave
from dataclasses import dataclass
from io import BytesIO
from os import getenv

import av

from app.services.audio_source import AudioSource

DEFAULT_MAX_DURATION_SECONDS = 120
PCM_SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_CHANNELS = 1
WAV_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}
MOBILE_AUDIO_CONTENT_TYPES = {
    "audio/aac",
    "audio/m4a",
    "audio/mp4",
    "audio/x-aac",
    "audio/x-m4a",
}
MOBILE_AUDIO_EXTENSIONS = {".aac", ".m4a", ".mp4"}


@dataclass(frozen=True)
class NormalizedAcousticAudio:
    data: bytes
    detected_signals: list[str]


class AudioNormalizationError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def normalize_audio_for_acoustic_detection(
    source: AudioSource,
) -> NormalizedAcousticAudio:
    if _looks_like_wav(source):
        return NormalizedAcousticAudio(data=source.data, detected_signals=[])

    if not _looks_like_mobile_audio(source):
        raise AudioNormalizationError("unsupported_format")

    try:
        pcm_wav = _decode_mobile_audio_to_pcm_wav(source.data)
    except AudioNormalizationError:
        raise
    except (EOFError, OSError, RuntimeError, ValueError, av.error.FFmpegError) as error:
        raise AudioNormalizationError("invalid_audio") from error

    return NormalizedAcousticAudio(
        data=pcm_wav,
        detected_signals=[
            "acoustic_normalization:mobile_audio_to_pcm_s16le",
            f"acoustic_normalization_sample_rate:{PCM_SAMPLE_RATE}",
        ],
    )


def _decode_mobile_audio_to_pcm_wav(data: bytes) -> bytes:
    source = BytesIO(data)
    output = BytesIO()
    max_samples = _get_max_duration_seconds() * PCM_SAMPLE_RATE
    decoded_samples = 0

    with av.open(source, mode="r") as container:
        if not container.streams.audio:
            raise AudioNormalizationError("audio_stream_missing")

        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=PCM_SAMPLE_RATE,
        )

        with wave.open(output, "wb") as pcm_wav:
            pcm_wav.setnchannels(PCM_CHANNELS)
            pcm_wav.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
            pcm_wav.setframerate(PCM_SAMPLE_RATE)

            for frame in container.decode(audio=0):
                for normalized_frame in resampler.resample(frame):
                    decoded_samples = _write_pcm_frame(
                        pcm_wav,
                        normalized_frame,
                        decoded_samples,
                        max_samples,
                    )

            for normalized_frame in resampler.resample(None):
                decoded_samples = _write_pcm_frame(
                    pcm_wav,
                    normalized_frame,
                    decoded_samples,
                    max_samples,
                )

    if decoded_samples == 0:
        raise AudioNormalizationError("audio_stream_empty")

    return output.getvalue()


def _write_pcm_frame(
    pcm_wav: wave.Wave_write,
    frame: av.AudioFrame,
    decoded_samples: int,
    max_samples: int,
) -> int:
    next_decoded_samples = decoded_samples + frame.samples
    if next_decoded_samples > max_samples:
        raise AudioNormalizationError("duration_limit_exceeded")

    expected_bytes = frame.samples * PCM_SAMPLE_WIDTH_BYTES * PCM_CHANNELS
    plane = bytes(frame.planes[0])
    if len(plane) < expected_bytes:
        raise AudioNormalizationError("invalid_decoded_audio")

    pcm_wav.writeframesraw(plane[:expected_bytes])
    return next_decoded_samples


def _looks_like_wav(source: AudioSource) -> bool:
    return (
        source.content_type.lower() in WAV_CONTENT_TYPES
        or source.filename.lower().endswith(".wav")
    )


def _looks_like_mobile_audio(source: AudioSource) -> bool:
    filename = source.filename.lower()

    return (
        source.content_type.lower() in MOBILE_AUDIO_CONTENT_TYPES
        or any(filename.endswith(extension) for extension in MOBILE_AUDIO_EXTENSIONS)
    )


def _get_max_duration_seconds() -> int:
    raw = getenv("AI_ACOUSTIC_MAX_DURATION_SECONDS")
    if not raw:
        return DEFAULT_MAX_DURATION_SECONDS

    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_DURATION_SECONDS

    return value if value > 0 else DEFAULT_MAX_DURATION_SECONDS
