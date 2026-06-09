from __future__ import annotations

import re
from dataclasses import dataclass
from unicodedata import category, normalize

from app.schemas.analyze import (
    AcousticEvent,
    AudioCaptureContext,
    AudioTranscription,
    RecommendedAction,
    RiskLevel,
    ThreatMatch,
    TranscriptionSegment,
)

MAX_THREAT_MATCHES = 12


@dataclass(frozen=True)
class TextRiskPattern:
    label: str
    severity: RiskLevel
    confidence: float
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class RiskAggregationResult:
    risk_level: RiskLevel
    confidence: float
    summary: str
    should_escalate: bool
    recommended_action: RecommendedAction
    threat_matches: list[ThreatMatch]
    detected_signals: list[str]


TEXT_RISK_PATTERNS = (
    TextRiskPattern(
        label="concrete_lethal_threat",
        severity=RiskLevel.CRITICAL,
        confidence=0.95,
        phrases=(
            "eu vou te matar",
            "vou te matar",
            "vou matar voce",
            "vou acabar com voce",
            "vou te esfaquear",
            "vou te dar um tiro",
            "vou atirar em voce",
            "vou te enforcar",
            "vou te queimar",
        ),
    ),
    TextRiskPattern(
        label="physical_assault_threat",
        severity=RiskLevel.HIGH,
        confidence=0.84,
        phrases=(
            "vou te bater",
            "vou te machucar",
            "vou te agredir",
            "voce vai apanhar",
            "vou te socar",
            "vou te dar um soco",
            "vou te arrebentar",
            "vou te quebrar",
        ),
    ),
    TextRiskPattern(
        label="active_distress_call",
        severity=RiskLevel.HIGH,
        confidence=0.82,
        phrases=(
            "socorro",
            "me ajuda",
            "me solta",
            "para de me bater",
            "nao me bate",
            "chama a policia",
        ),
    ),
    TextRiskPattern(
        label="severe_verbal_abuse",
        severity=RiskLevel.MEDIUM,
        confidence=0.64,
        phrases=(
            "sua vagabunda",
            "vagabunda",
            "sua puta",
            "puta",
            "piranha",
            "desgracada",
            "cala a boca",
        ),
    ),
)

RISK_RANK = {
    RiskLevel.UNKNOWN: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


def aggregate_risk(
    transcription: AudioTranscription | None,
    acoustic_events: list[AcousticEvent],
    capture_context: AudioCaptureContext | None,
) -> RiskAggregationResult:
    threat_matches = _detect_text_threats(transcription)
    acoustic_risk = _get_acoustic_risk(acoustic_events)
    context_risk = _get_context_risk(capture_context)
    risk_level = _max_risk(
        RiskLevel.LOW,
        acoustic_risk,
        context_risk,
        *(match.severity for match in threat_matches),
    )

    if _should_promote_to_critical(threat_matches, acoustic_events, context_risk):
        risk_level = RiskLevel.CRITICAL

    confidence = _get_confidence(
        risk_level,
        threat_matches,
        acoustic_events,
        capture_context,
    )
    should_escalate = risk_level == RiskLevel.CRITICAL
    detected_signals = _get_detected_signals(
        risk_level,
        threat_matches,
        acoustic_events,
        capture_context,
    )

    return RiskAggregationResult(
        risk_level=risk_level,
        confidence=confidence,
        summary=_get_summary(
            risk_level=risk_level,
            threat_matches=threat_matches,
            acoustic_events=acoustic_events,
            has_transcription=transcription is not None,
        ),
        should_escalate=should_escalate,
        recommended_action=_get_recommended_action(risk_level),
        threat_matches=threat_matches,
        detected_signals=detected_signals,
    )


def _detect_text_threats(
    transcription: AudioTranscription | None,
) -> list[ThreatMatch]:
    if transcription is None or not transcription.text.strip():
        return []

    normalized_text = _normalize_text(transcription.text)
    matches: list[ThreatMatch] = []
    seen: set[tuple[str, str]] = set()

    for pattern in TEXT_RISK_PATTERNS:
        for phrase in pattern.phrases:
            normalized_phrase = _normalize_text(phrase)
            match_index = _find_phrase_index(normalized_text, normalized_phrase)
            if match_index < 0 or _is_negated(normalized_text, match_index):
                continue

            key = (pattern.label, normalized_phrase)
            if key in seen:
                continue

            segment = _find_matching_segment(transcription, normalized_phrase)
            matches.append(
                ThreatMatch(
                    label=pattern.label,
                    severity=pattern.severity,
                    confidence=pattern.confidence,
                    start_ms=segment.start_ms if segment else None,
                    end_ms=segment.end_ms if segment else None,
                    evidence=phrase,
                )
            )
            seen.add(key)

            if len(matches) >= MAX_THREAT_MATCHES:
                return matches

    return matches


def _get_acoustic_risk(acoustic_events: list[AcousticEvent]) -> RiskLevel:
    if any(
        event.label in {"impact_candidate", "clipping_detected"}
        for event in acoustic_events
    ):
        return RiskLevel.HIGH

    if acoustic_events:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def _get_context_risk(capture_context: AudioCaptureContext | None) -> RiskLevel:
    if not capture_context:
        return RiskLevel.LOW

    critical_triggers = {
        "critical_alert",
        "emergency_button",
        "manual_sos",
        "panic_button",
    }
    high_triggers = {
        "impact",
        "local_critical_candidate",
        "scream",
    }
    medium_triggers = {
        "sustained_loud_audio",
        "very_loud_audio",
        "volume_spike",
    }
    trigger_reasons = {
        _normalize_text(reason)
        for reason in capture_context.trigger_reasons
    }

    if trigger_reasons & critical_triggers:
        return RiskLevel.CRITICAL

    if trigger_reasons & high_triggers:
        return RiskLevel.HIGH

    if trigger_reasons & medium_triggers:
        return RiskLevel.MEDIUM

    return RiskLevel.LOW


def _should_promote_to_critical(
    threat_matches: list[ThreatMatch],
    acoustic_events: list[AcousticEvent],
    context_risk: RiskLevel,
) -> bool:
    if context_risk == RiskLevel.CRITICAL:
        return True

    if any(match.severity == RiskLevel.CRITICAL for match in threat_matches):
        return True

    has_high_text_signal = any(
        match.severity == RiskLevel.HIGH
        for match in threat_matches
    )
    has_critical_acoustic_signal = any(
        event.label in {"impact_candidate", "clipping_detected"}
        for event in acoustic_events
    )

    return has_high_text_signal and has_critical_acoustic_signal


def _get_confidence(
    risk_level: RiskLevel,
    threat_matches: list[ThreatMatch],
    acoustic_events: list[AcousticEvent],
    capture_context: AudioCaptureContext | None,
) -> float:
    scores = [
        match.confidence for match in threat_matches
    ] + [
        event.confidence for event in acoustic_events
    ]

    if capture_context and capture_context.local_confidence is not None:
        scores.append(capture_context.local_confidence)

    if not scores:
        return 0.12 if risk_level == RiskLevel.LOW else 0

    return round(min(0.99, max(scores)), 3)


def _get_detected_signals(
    risk_level: RiskLevel,
    threat_matches: list[ThreatMatch],
    acoustic_events: list[AcousticEvent],
    capture_context: AudioCaptureContext | None,
) -> list[str]:
    signals = [
        "risk_aggregation_completed",
        f"risk_level:{risk_level}",
    ]

    signals.extend(
        f"threat_signal:{match.label}"
        for match in threat_matches
    )

    if any(
        event.label in {"impact_candidate", "clipping_detected"}
        for event in acoustic_events
    ):
        signals.append("risk_input:acoustic_critical_candidate")
    elif acoustic_events:
        signals.append("risk_input:acoustic_relevant")

    if capture_context and capture_context.location:
        signals.append("risk_context:location_present")

    if capture_context and capture_context.trigger_reasons:
        signals.extend(
            f"risk_context_trigger:{_normalize_text(reason)}"
            for reason in capture_context.trigger_reasons[:5]
        )

    return signals


def _get_summary(
    risk_level: RiskLevel,
    threat_matches: list[ThreatMatch],
    acoustic_events: list[AcousticEvent],
    has_transcription: bool,
) -> str:
    has_text_signal = bool(threat_matches)
    has_critical_acoustic_signal = any(
        event.label in {"impact_candidate", "clipping_detected"}
        for event in acoustic_events
    )

    if risk_level == RiskLevel.CRITICAL:
        if has_text_signal and has_critical_acoustic_signal:
            return (
                "Critical risk candidate detected from threat language and "
                "impact-like acoustic signals. Escalation to emergency contacts "
                "is recommended."
            )

        if has_text_signal:
            return (
                "Critical risk candidate detected from concrete threat language. "
                "Escalation to emergency contacts is recommended."
            )

        return (
            "Critical risk candidate detected from capture context. Escalation "
            "to emergency contacts is recommended."
        )

    if risk_level == RiskLevel.HIGH:
        return (
            "High-risk evidence candidate detected. Store the clip and request "
            "human review before any contact escalation."
        )

    if risk_level == RiskLevel.MEDIUM:
        return (
            "Relevant evidence candidate detected. Store the clip with transcript "
            "and acoustic metadata for later review."
        )

    if has_transcription:
        return (
            "Audio transcription and acoustic analysis completed without threat "
            "signals in this first-pass classifier."
        )

    return (
        "Mock analysis completed. No real model was executed and no critical "
        "escalation was inferred from metadata-only input."
    )


def _get_recommended_action(risk_level: RiskLevel) -> RecommendedAction:
    if risk_level == RiskLevel.CRITICAL:
        return RecommendedAction.ESCALATE_CONTACTS

    if risk_level == RiskLevel.HIGH:
        return RecommendedAction.REVIEW

    if risk_level == RiskLevel.MEDIUM:
        return RecommendedAction.STORE_EVIDENCE

    return RecommendedAction.NONE


def _max_risk(*risk_levels: RiskLevel) -> RiskLevel:
    return max(risk_levels, key=lambda level: RISK_RANK[level])


def _find_matching_segment(
    transcription: AudioTranscription,
    normalized_phrase: str,
) -> TranscriptionSegment | None:
    for segment in transcription.segments:
        if _find_phrase_index(_normalize_text(segment.text), normalized_phrase) >= 0:
            return segment

    return None


def _is_negated(normalized_text: str, match_index: int) -> bool:
    previous = normalized_text[max(0, match_index - 12):match_index].strip()

    return previous.endswith("nao") or previous.endswith("nunca")


def _find_phrase_index(normalized_text: str, normalized_phrase: str) -> int:
    pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
    match = re.search(pattern, normalized_text)

    return match.start() if match else -1


def _normalize_text(value: str) -> str:
    stripped_accents = "".join(
        char
        for char in normalize("NFKD", value)
        if category(char) != "Mn"
    )

    return " ".join(stripped_accents.lower().split())
