from __future__ import annotations

import argparse
import base64
import json
import math
import os
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from struct import pack
from time import perf_counter
from typing import Iterator

from app.schemas.analyze import (
    AnalyzeEvidenceRequest,
    EvidenceType,
    RecommendedAction,
    RiskLevel,
)
from app.services.acoustic_detection_service import (
    DEFAULT_CLIPPING_RATIO_THRESHOLD,
    DEFAULT_HIGH_RMS_THRESHOLD,
    DEFAULT_IMPACT_DELTA_THRESHOLD,
    DEFAULT_PEAK_THRESHOLD,
    DEFAULT_WINDOW_MS,
)
from app.services.analyze_service import AnalyzeService

SAMPLE_RATE = 16_000
DEFAULT_DURATION_MS = 1_200
LOCAL_FIXTURE_EXTENSIONS = {
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}
RISK_RANK = {
    RiskLevel.UNKNOWN: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    title: str
    description: str
    mock_transcript: str
    audio_profile: str
    expected_min_risk: RiskLevel
    expected_max_risk: RiskLevel
    expected_should_escalate: bool
    expected_action: RecommendedAction
    trigger_reasons: tuple[str, ...] = ()
    local_confidence: float = 0.32


@dataclass(frozen=True)
class EvaluationAudio:
    data: bytes
    mime_type: str
    duration_ms: int
    source_kind: str


EVALUATION_CASES = (
    EvaluationCase(
        id="benign_quiet",
        title="Benign quiet conversation",
        description="Neutral transcript and quiet synthetic audio should stay LOW.",
        mock_transcript="Hoje foi um dia tranquilo. Vou preparar o jantar.",
        audio_profile="quiet",
        expected_min_risk=RiskLevel.LOW,
        expected_max_risk=RiskLevel.LOW,
        expected_should_escalate=False,
        expected_action=RecommendedAction.NONE,
    ),
    EvaluationCase(
        id="tense_loud_no_threat",
        title="Tense loud audio without threat language",
        description=(
            "Raised-volume synthetic audio without threat words should be stored "
            "for review, not escalated."
        ),
        mock_transcript="Para, vamos conversar com calma. Eu preciso sair daqui.",
        audio_profile="loud",
        expected_min_risk=RiskLevel.MEDIUM,
        expected_max_risk=RiskLevel.MEDIUM,
        expected_should_escalate=False,
        expected_action=RecommendedAction.STORE_EVIDENCE,
        local_confidence=0.61,
    ),
    EvaluationCase(
        id="severe_verbal_abuse",
        title="Severe verbal abuse",
        description="Insult language should become relevant evidence but not critical.",
        mock_transcript="Cala a boca, sua vagabunda.",
        audio_profile="voice_like",
        expected_min_risk=RiskLevel.MEDIUM,
        expected_max_risk=RiskLevel.MEDIUM,
        expected_should_escalate=False,
        expected_action=RecommendedAction.STORE_EVIDENCE,
    ),
    EvaluationCase(
        id="physical_assault_threat",
        title="Physical assault threat",
        description="Concrete physical threat should become HIGH without auto dispatch.",
        mock_transcript="Se voce continuar falando, vou te bater.",
        audio_profile="voice_like",
        expected_min_risk=RiskLevel.HIGH,
        expected_max_risk=RiskLevel.HIGH,
        expected_should_escalate=False,
        expected_action=RecommendedAction.REVIEW,
    ),
    EvaluationCase(
        id="lethal_threat",
        title="Concrete lethal threat",
        description="Concrete lethal threat should trigger CRITICAL.",
        mock_transcript="Eu vou te matar agora.",
        audio_profile="voice_like",
        expected_min_risk=RiskLevel.CRITICAL,
        expected_max_risk=RiskLevel.CRITICAL,
        expected_should_escalate=True,
        expected_action=RecommendedAction.ESCALATE_CONTACTS,
    ),
    EvaluationCase(
        id="distress_with_impact",
        title="Distress call with impact-like audio",
        description="Distress language plus impact-like signal should trigger CRITICAL.",
        mock_transcript="Socorro, me ajuda, para de me bater.",
        audio_profile="impact",
        expected_min_risk=RiskLevel.CRITICAL,
        expected_max_risk=RiskLevel.CRITICAL,
        expected_should_escalate=True,
        expected_action=RecommendedAction.ESCALATE_CONTACTS,
        trigger_reasons=("impact", "scream"),
        local_confidence=0.84,
    ),
)


def run_evaluation(
    *,
    provider_mode: str = "mock",
    fixtures_dir: Path | None = None,
    cost_usd_per_hour: float | None = None,
) -> dict:
    generated_at = datetime.now(UTC)
    cases = [
        _run_case(
            case,
            provider_mode=provider_mode,
            fixtures_dir=fixtures_dir,
        )
        for case in EVALUATION_CASES
    ]
    failed_cases = [case for case in cases if not case["passed"]]
    false_positives = [
        case
        for case in cases
        if case["actualRiskRank"] > case["expectedMaxRiskRank"]
        or (
            case["actualShouldEscalate"]
            and not case["expectedShouldEscalate"]
        )
    ]
    false_negatives = [
        case
        for case in cases
        if case["actualRiskRank"] < case["expectedMinRiskRank"]
        or (
            case["expectedShouldEscalate"]
            and not case["actualShouldEscalate"]
        )
    ]
    total_audio_seconds = round(
        sum(case["audioDurationMs"] for case in cases) / 1000,
        3,
    )
    latency_values = [case["latencyMs"] for case in cases]
    wall_latency_values = [case["wallLatencyMs"] for case in cases]
    estimated_cost_usd = (
        round((total_audio_seconds / 3600) * cost_usd_per_hour, 6)
        if cost_usd_per_hour is not None
        else None
    )

    return {
        "generatedAt": generated_at.isoformat(),
        "providerMode": provider_mode,
        "configuredProvider": os.getenv("AI_TRANSCRIPTION_PROVIDER"),
        "configuredProviderChain": os.getenv("AI_TRANSCRIPTION_PROVIDER_CHAIN"),
        "caseCount": len(cases),
        "passed": len(cases) - len(failed_cases),
        "failed": len(failed_cases),
        "falsePositiveCount": len(false_positives),
        "falseNegativeCount": len(false_negatives),
        "totalAudioSeconds": total_audio_seconds,
        "estimatedCostUsd": estimated_cost_usd,
        "costUsdPerHour": cost_usd_per_hour,
        "latencyMs": {
            "average": _average(latency_values),
            "max": max(latency_values, default=0),
        },
        "wallLatencyMs": {
            "average": _average(wall_latency_values),
            "max": max(wall_latency_values, default=0),
        },
        "thresholds": _get_thresholds(),
        "limitations": [
            "Public CI fixtures are synthetic and do not represent real victims, "
            "real attackers, accents, noisy homes, or legal evidence quality.",
            "Mock mode validates the deterministic risk policy and acoustic "
            "heuristics; it does not validate provider transcription accuracy.",
            "Configured-provider mode should use consented local fixtures kept "
            "outside git, and provider pricing must be supplied or checked in "
            "the provider dashboard before relying on cost estimates.",
        ],
        "cases": cases,
    }


def render_markdown_report(report: dict) -> str:
    lines = [
        "# Vera AI Audio Evaluation",
        "",
        f"- Generated at: `{report['generatedAt']}`",
        f"- Provider mode: `{report['providerMode']}`",
        f"- Configured provider: `{report.get('configuredProvider') or 'unset'}`",
        f"- Configured chain: `{report.get('configuredProviderChain') or 'unset'}`",
        f"- Cases: `{report['passed']}/{report['caseCount']}` passed",
        f"- False positives: `{report['falsePositiveCount']}`",
        f"- False negatives: `{report['falseNegativeCount']}`",
        f"- Total audio seconds: `{report['totalAudioSeconds']}`",
        f"- Estimated cost USD: `{report['estimatedCostUsd']}`",
        f"- Latency ms avg/max: `{report['latencyMs']['average']}` / "
        f"`{report['latencyMs']['max']}`",
        "",
        "## Thresholds",
        "",
        "- Acoustic window ms: "
        f"`{report['thresholds']['acoustic']['windowMs']}`",
        "- Acoustic high RMS threshold: "
        f"`{report['thresholds']['acoustic']['highRmsThreshold']}`",
        "- Acoustic peak threshold: "
        f"`{report['thresholds']['acoustic']['peakThreshold']}`",
        "- Acoustic impact delta threshold: "
        f"`{report['thresholds']['acoustic']['impactDeltaThreshold']}`",
        "- Acoustic clipping ratio threshold: "
        f"`{report['thresholds']['acoustic']['clippingRatioThreshold']}`",
        "- Critical policy: "
        f"{'; '.join(report['thresholds']['riskPolicy']['criticalRules'])}",
        "",
        "## Cases",
        "",
    ]

    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        lines.extend(
            [
                f"### {status} - {case['id']}",
                "",
                f"- Title: {case['title']}",
                f"- Expected risk: {case['expectedMinRisk']}.."
                f"{case['expectedMaxRisk']}",
                f"- Actual risk: {case['actualRisk']}",
                f"- Expected escalation: `{case['expectedShouldEscalate']}`",
                f"- Actual escalation: `{case['actualShouldEscalate']}`",
                f"- Recommended action: `{case['actualRecommendedAction']}`",
                f"- Provider: `{case['provider']}` / `{case['model']}`",
                f"- Audio source: `{case['audioSourceKind']}`",
                f"- Latency ms: `{case['latencyMs']}`",
                f"- Failure reason: `{case['failureReason'] or 'none'}`",
                "",
            ]
        )

    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Vera AI audio evaluation harness.",
    )
    parser.add_argument(
        "--provider-mode",
        choices=("mock", "configured"),
        default="mock",
        help=(
            "mock uses synthetic transcripts for CI; configured uses the current "
            "AI_TRANSCRIPTION_PROVIDER or AI_TRANSCRIPTION_PROVIDER_CHAIN env."
        ),
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        help=(
            "Optional local directory with consented audio files named after case "
            "ids, for example lethal_threat.m4a. These files should stay out of git."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional report file path.",
    )
    parser.add_argument(
        "--cost-usd-per-hour",
        type=float,
        default=_get_env_float_or_none("AI_EVALUATION_COST_USD_PER_HOUR"),
        help="Optional provider cost used to estimate run cost.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even when one or more cases fail.",
    )

    args = parser.parse_args()
    report = run_evaluation(
        provider_mode=args.provider_mode,
        fixtures_dir=args.fixtures_dir,
        cost_usd_per_hour=args.cost_usd_per_hour,
    )
    output = (
        json.dumps(report, indent=2, ensure_ascii=False)
        if args.format == "json"
        else render_markdown_report(report)
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)

    if report["failed"] and not args.allow_failures:
        raise SystemExit(1)


def _run_case(
    case: EvaluationCase,
    *,
    provider_mode: str,
    fixtures_dir: Path | None,
) -> dict:
    audio = _load_case_audio(case, fixtures_dir)
    payload = _build_payload(case, audio)
    updates: dict[str, str] = {}
    deletes: tuple[str, ...] = ()

    if provider_mode == "mock":
        updates = {
            "AI_TRANSCRIPTION_PROVIDER": "mock",
            "AI_MOCK_TRANSCRIPTION_TEXT": case.mock_transcript,
        }
        deletes = ("AI_TRANSCRIPTION_PROVIDER_CHAIN",)

    started = perf_counter()
    with _temporary_env(updates=updates, deletes=deletes):
        response = AnalyzeService().analyze(payload)
    wall_latency_ms = round((perf_counter() - started) * 1000)

    actual_risk = response.risk_level
    passed = (
        RISK_RANK[case.expected_min_risk]
        <= RISK_RANK[actual_risk]
        <= RISK_RANK[case.expected_max_risk]
        and response.should_escalate == case.expected_should_escalate
        and response.recommended_action == case.expected_action
    )

    return {
        "id": case.id,
        "title": case.title,
        "description": case.description,
        "passed": passed,
        "expectedMinRisk": case.expected_min_risk.value,
        "expectedMaxRisk": case.expected_max_risk.value,
        "expectedMinRiskRank": RISK_RANK[case.expected_min_risk],
        "expectedMaxRiskRank": RISK_RANK[case.expected_max_risk],
        "actualRisk": actual_risk.value,
        "actualRiskRank": RISK_RANK[actual_risk],
        "expectedShouldEscalate": case.expected_should_escalate,
        "actualShouldEscalate": response.should_escalate,
        "expectedRecommendedAction": case.expected_action.value,
        "actualRecommendedAction": response.recommended_action.value,
        "provider": response.provider_metadata.provider,
        "model": response.provider_metadata.model,
        "status": response.status.value,
        "confidence": response.confidence,
        "summary": response.summary,
        "failureReason": response.failure_reason,
        "detectedSignals": response.detected_signals,
        "acousticEvents": [
            event.model_dump(mode="json", by_alias=True)
            for event in response.acoustic_events
        ],
        "threatMatches": [
            match.model_dump(mode="json", by_alias=True)
            for match in response.threat_matches
        ],
        "latencyMs": response.latency_ms,
        "wallLatencyMs": wall_latency_ms,
        "audioDurationMs": audio.duration_ms,
        "audioSourceKind": audio.source_kind,
    }


def _build_payload(
    case: EvaluationCase,
    audio: EvaluationAudio,
) -> AnalyzeEvidenceRequest:
    encoded_audio = base64.b64encode(audio.data).decode("ascii")
    capture_started_at = datetime(2026, 6, 3, 12, 0, tzinfo=UTC)
    capture_ended_at = capture_started_at + timedelta(milliseconds=audio.duration_ms)

    return AnalyzeEvidenceRequest(
        evidence_record_id=f"eval-{case.id}",
        alert_session_id="eval-session",
        evidence_type=EvidenceType.AUDIO,
        mime_type=audio.mime_type,
        size=len(audio.data),
        content_hash=sha256(audio.data).hexdigest(),
        storage_reference=f"data:{audio.mime_type};base64,{encoded_audio}",
        capture_context={
            "capture_started_at": capture_started_at,
            "capture_ended_at": capture_ended_at,
            "triggered_at": capture_started_at + timedelta(milliseconds=300),
            "pre_roll_ms": 300,
            "post_roll_ms": 700,
            "trigger_reasons": list(case.trigger_reasons),
            "local_confidence": case.local_confidence,
            "platform": "evaluation",
            "foreground": True,
        },
    )


def _load_case_audio(
    case: EvaluationCase,
    fixtures_dir: Path | None,
) -> EvaluationAudio:
    if fixtures_dir:
        for extension, mime_type in LOCAL_FIXTURE_EXTENSIONS.items():
            path = fixtures_dir / f"{case.id}{extension}"
            if path.is_file():
                data = path.read_bytes()
                return EvaluationAudio(
                    data=data,
                    mime_type=mime_type,
                    duration_ms=_get_wav_duration_ms(data) or DEFAULT_DURATION_MS,
                    source_kind="local_fixture",
                )

    return EvaluationAudio(
        data=_build_synthetic_wav(case.audio_profile, DEFAULT_DURATION_MS),
        mime_type="audio/wav",
        duration_ms=DEFAULT_DURATION_MS,
        source_kind="synthetic_public",
    )


def _build_synthetic_wav(profile: str, duration_ms: int) -> bytes:
    sample_count = round(SAMPLE_RATE * duration_ms / 1000)
    samples: list[int] = []

    for index in range(sample_count):
        time_seconds = index / SAMPLE_RATE
        window_start_ms = math.floor((index / SAMPLE_RATE) * 1000)

        if profile == "quiet":
            sample = 0
        elif profile == "voice_like":
            sample = round(2200 * math.sin(2 * math.pi * 220 * time_seconds))
        elif profile == "loud":
            sample = round(28500 * math.sin(2 * math.pi * 280 * time_seconds))
        elif profile == "impact":
            if 400 <= window_start_ms < 500:
                sample = round(29200 * math.sin(2 * math.pi * 240 * time_seconds))
            else:
                sample = round(1200 * math.sin(2 * math.pi * 180 * time_seconds))
        else:
            sample = 0

        samples.append(max(-32768, min(32767, sample)))

    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(b"".join(pack("<h", sample) for sample in samples))

    return buffer.getvalue()


def _get_wav_duration_ms(data: bytes) -> int | None:
    try:
        with wave.open(BytesIO(data), "rb") as audio:
            return round(audio.getnframes() * 1000 / audio.getframerate())
    except wave.Error:
        return None


@contextmanager
def _temporary_env(
    *,
    updates: dict[str, str],
    deletes: tuple[str, ...],
) -> Iterator[None]:
    names = set(updates) | set(deletes)
    previous = {name: os.environ.get(name) for name in names}

    try:
        for name in deletes:
            os.environ.pop(name, None)

        for name, value in updates.items():
            os.environ[name] = value

        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _average(values: list[int]) -> float:
    if not values:
        return 0

    return round(sum(values) / len(values), 3)


def _get_env_float_or_none(name: str) -> float | None:
    raw = os.getenv(name)
    if not raw:
        return None

    try:
        value = float(raw)
    except ValueError:
        return None

    return value if value >= 0 else None


def _get_thresholds() -> dict:
    return {
        "acoustic": {
            "windowMs": _get_env_int_or_default(
                "AI_ACOUSTIC_WINDOW_MS",
                DEFAULT_WINDOW_MS,
            ),
            "highRmsThreshold": _get_env_float_or_default(
                "AI_ACOUSTIC_HIGH_RMS_THRESHOLD",
                DEFAULT_HIGH_RMS_THRESHOLD,
            ),
            "peakThreshold": _get_env_float_or_default(
                "AI_ACOUSTIC_PEAK_THRESHOLD",
                DEFAULT_PEAK_THRESHOLD,
            ),
            "impactDeltaThreshold": _get_env_float_or_default(
                "AI_ACOUSTIC_IMPACT_DELTA_THRESHOLD",
                DEFAULT_IMPACT_DELTA_THRESHOLD,
            ),
            "clippingRatioThreshold": _get_env_float_or_default(
                "AI_ACOUSTIC_CLIPPING_RATIO_THRESHOLD",
                DEFAULT_CLIPPING_RATIO_THRESHOLD,
            ),
        },
        "riskPolicy": {
            "mediumRules": [
                "severe verbal abuse text",
                "non-critical acoustic event",
            ],
            "highRules": [
                "physical assault threat text",
                "distress call text",
                "impact/clipping acoustic event",
                "high-risk local trigger",
            ],
            "criticalRules": [
                "concrete lethal threat text",
                "critical local trigger",
                "high-risk text plus impact/clipping acoustic event",
            ],
        },
    }


def _get_env_float_or_default(name: str, default: float) -> float:
    value = _get_env_float_or_none(name)

    return default if value is None else value


def _get_env_int_or_default(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default

    return value if value > 0 else default


if __name__ == "__main__":
    main()
