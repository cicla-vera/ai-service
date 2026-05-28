from __future__ import annotations

import wave
from dataclasses import dataclass
from io import BytesIO
from math import sqrt
from os import getenv
from struct import unpack

from app.schemas.analyze import AcousticEvent
from app.services.audio_source import AudioSource

DEFAULT_WINDOW_MS = 100
DEFAULT_HIGH_RMS_THRESHOLD = 0.42
DEFAULT_PEAK_THRESHOLD = 0.82
DEFAULT_IMPACT_DELTA_THRESHOLD = 0.35
DEFAULT_CLIPPING_RATIO_THRESHOLD = 0.03
MAX_ACOUSTIC_EVENTS = 20


@dataclass(frozen=True)
class AcousticDetectionResult:
    events: list[AcousticEvent]
    detected_signals: list[str]
    confidence: float


@dataclass(frozen=True)
class AudioWindow:
    start_ms: int
    end_ms: int
    rms: float
    peak: float
    clipping_ratio: float


def detect_acoustic_events(source: AudioSource | None) -> AcousticDetectionResult:
    if source is None:
        return AcousticDetectionResult(
            events=[],
            detected_signals=["acoustic_detection_skipped:no_audio_source"],
            confidence=0,
        )

    if not _looks_like_wav(source):
        return AcousticDetectionResult(
            events=[],
            detected_signals=["acoustic_detection_skipped:unsupported_format"],
            confidence=0,
        )

    try:
        windows = _read_pcm_windows(source.data)
    except (EOFError, ValueError, wave.Error):
        return AcousticDetectionResult(
            events=[],
            detected_signals=["acoustic_detection_skipped:invalid_wav"],
            confidence=0,
        )

    events = _detect_events(windows)
    signals = ["acoustic_detection_completed"]

    if events:
        signals.append("acoustic_signal:relevant")

    if any(event.label in {"impact_candidate", "clipping_detected"} for event in events):
        signals.append("acoustic_signal:critical_candidate")

    return AcousticDetectionResult(
        events=events,
        detected_signals=signals,
        confidence=max((event.confidence for event in events), default=0),
    )


def _looks_like_wav(source: AudioSource) -> bool:
    return (
        source.content_type.lower() in {"audio/wav", "audio/x-wav", "audio/wave"}
        or source.filename.lower().endswith(".wav")
    )


def _read_pcm_windows(data: bytes) -> list[AudioWindow]:
    with wave.open(BytesIO(data), "rb") as audio:
        sample_width = audio.getsampwidth()
        if sample_width not in {1, 2, 4}:
            raise ValueError("unsupported_sample_width")

        channels = audio.getnchannels()
        frame_rate = audio.getframerate()
        window_frames = max(1, round(frame_rate * _window_ms() / 1000))
        windows: list[AudioWindow] = []
        frame_cursor = 0

        while True:
            frames = audio.readframes(window_frames)
            if not frames:
                break

            samples = _decode_pcm_samples(frames, sample_width)
            if channels > 1:
                samples = samples[::channels]

            start_ms = round(frame_cursor * 1000 / frame_rate)
            frame_count = len(samples)
            end_ms = round((frame_cursor + frame_count) * 1000 / frame_rate)
            frame_cursor += frame_count

            windows.append(
                AudioWindow(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    rms=_rms(samples, sample_width),
                    peak=_peak(samples, sample_width),
                    clipping_ratio=_clipping_ratio(samples, sample_width),
                )
            )

        return windows


def _decode_pcm_samples(frames: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [sample - 128 for sample in frames]

    if sample_width == 2:
        count = len(frames) // 2
        return list(unpack(f"<{count}h", frames))

    count = len(frames) // 4
    return list(unpack(f"<{count}i", frames))


def _rms(samples: list[int], sample_width: int) -> float:
    if not samples:
        return 0

    total = sum(sample * sample for sample in samples)
    return min(1, sqrt(total / len(samples)) / _max_amplitude(sample_width))


def _peak(samples: list[int], sample_width: int) -> float:
    if not samples:
        return 0

    return min(1, max(abs(sample) for sample in samples) / _max_amplitude(sample_width))


def _clipping_ratio(samples: list[int], sample_width: int) -> float:
    if not samples:
        return 0

    max_amplitude = _max_amplitude(sample_width)
    clipped = sum(1 for sample in samples if abs(sample) >= max_amplitude * 0.98)

    return clipped / len(samples)


def _max_amplitude(sample_width: int) -> int:
    if sample_width == 1:
        return 128

    return (2 ** (sample_width * 8)) // 2


def _detect_events(windows: list[AudioWindow]) -> list[AcousticEvent]:
    events: list[AcousticEvent] = []
    high_rms_threshold = _env_float(
        "AI_ACOUSTIC_HIGH_RMS_THRESHOLD",
        DEFAULT_HIGH_RMS_THRESHOLD,
    )
    peak_threshold = _env_float("AI_ACOUSTIC_PEAK_THRESHOLD", DEFAULT_PEAK_THRESHOLD)
    clipping_ratio_threshold = _env_float(
        "AI_ACOUSTIC_CLIPPING_RATIO_THRESHOLD",
        DEFAULT_CLIPPING_RATIO_THRESHOLD,
    )

    for index, window in enumerate(windows):
        if window.rms >= high_rms_threshold and window.peak >= peak_threshold:
            events.append(
                _event(
                    label="sustained_loud_audio",
                    window=window,
                    confidence=max(window.rms, window.peak),
                )
            )

        if window.clipping_ratio >= clipping_ratio_threshold:
            events.append(
                _event(
                    label="clipping_detected",
                    window=window,
                    confidence=min(1, window.clipping_ratio / clipping_ratio_threshold),
                )
            )

        if index > 0 and _is_impact_candidate(windows[index - 1], window):
            events.append(
                _event(
                    label="impact_candidate",
                    window=window,
                    confidence=max(window.peak, window.rms),
                )
            )

        if len(events) >= MAX_ACOUSTIC_EVENTS:
            return events[:MAX_ACOUSTIC_EVENTS]

    return events


def _is_impact_candidate(previous: AudioWindow, current: AudioWindow) -> bool:
    impact_delta_threshold = _env_float(
        "AI_ACOUSTIC_IMPACT_DELTA_THRESHOLD",
        DEFAULT_IMPACT_DELTA_THRESHOLD,
    )
    peak_threshold = _env_float("AI_ACOUSTIC_PEAK_THRESHOLD", DEFAULT_PEAK_THRESHOLD)

    return (
        current.peak >= peak_threshold
        and current.rms - previous.rms >= impact_delta_threshold
    )


def _event(label: str, window: AudioWindow, confidence: float) -> AcousticEvent:
    return AcousticEvent(
        label=label,
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        confidence=round(min(1, max(0, confidence)), 3),
        source="heuristic:wav_pcm",
    )


def _window_ms() -> int:
    raw = getenv("AI_ACOUSTIC_WINDOW_MS")
    if not raw:
        return DEFAULT_WINDOW_MS

    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_WINDOW_MS

    return value if value > 0 else DEFAULT_WINDOW_MS


def _env_float(name: str, default: float) -> float:
    raw = getenv(name)
    if not raw:
        return default

    try:
        value = float(raw)
    except ValueError:
        return default

    return value if 0 <= value <= 1 else default
