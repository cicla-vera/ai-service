from datetime import UTC, datetime

from app.schemas.analyze import (
    AnalysisProviderMetadata,
    AnalysisStatus,
    AnalyzeEvidenceRequest,
    AnalyzeEvidenceResponse,
    EvidenceWindow,
    RecommendedAction,
    RiskLevel,
)
from app.services.acoustic_detection_service import detect_acoustic_events
from app.services.audio_source import AudioSourceError, resolve_audio_source
from app.services.risk_aggregation_service import aggregate_risk
from app.services.transcription_service import (
    TranscriptionError,
    get_transcription_provider,
)

ANALYSIS_VERSION = "audio-evidence-v1"


class AnalyzeService:
    def analyze(self, payload: AnalyzeEvidenceRequest) -> AnalyzeEvidenceResponse:
        processing_started_at = datetime.now(UTC)

        try:
            source = resolve_audio_source(payload)
            acoustic_result = detect_acoustic_events(source)
            transcription_provider = get_transcription_provider()
            transcription_result = transcription_provider.transcribe(source, payload)
        except AudioSourceError as error:
            return self._build_failed_response(
                payload=payload,
                processing_started_at=processing_started_at,
                failure_reason=error.code,
            )
        except TranscriptionError as error:
            return self._build_failed_response(
                payload=payload,
                processing_started_at=processing_started_at,
                failure_reason=error.code,
                detected_signals=error.detected_signals,
            )

        processing_finished_at = datetime.now(UTC)
        detected_signals = [
            "metadata_received",
            f"evidence_type:{payload.evidence_type}",
            *transcription_result.detected_signals,
            *acoustic_result.detected_signals,
        ]

        if transcription_result.transcription is None:
            detected_signals.insert(0, "mock_analysis")

        risk_result = aggregate_risk(
            transcription=transcription_result.transcription,
            acoustic_events=acoustic_result.events,
            capture_context=payload.capture_context,
        )
        detected_signals.extend(risk_result.detected_signals)

        return AnalyzeEvidenceResponse(
            analysis_id=(
                f"{transcription_result.provider_metadata.provider}-analysis-"
                f"{payload.evidence_record_id}"
            ),
            analysis_version=ANALYSIS_VERSION,
            status=transcription_result.status,
            risk_level=risk_result.risk_level,
            confidence=max(
                transcription_result.confidence,
                acoustic_result.confidence,
                risk_result.confidence,
            ),
            summary=risk_result.summary,
            detected_signals=detected_signals,
            should_escalate=risk_result.should_escalate,
            recommended_action=risk_result.recommended_action,
            evidence_window=self._get_evidence_window(payload),
            transcription=transcription_result.transcription,
            acoustic_events=acoustic_result.events,
            threat_matches=risk_result.threat_matches,
            provider_metadata=transcription_result.provider_metadata,
            processing_started_at=processing_started_at,
            processing_finished_at=processing_finished_at,
            latency_ms=self._get_latency_ms(
                processing_started_at,
                processing_finished_at,
            ),
            failure_reason=transcription_result.failure_reason,
        )

    def _get_evidence_window(self, payload: AnalyzeEvidenceRequest) -> EvidenceWindow:
        context = payload.capture_context

        if not context:
            return EvidenceWindow(started_at=None, ended_at=None, duration_ms=None)

        duration_ms = None
        if context.capture_started_at and context.capture_ended_at:
            duration_ms = max(
                0,
                round(
                    (
                        context.capture_ended_at - context.capture_started_at
                    ).total_seconds()
                    * 1000,
                ),
            )

        return EvidenceWindow(
            started_at=context.capture_started_at,
            ended_at=context.capture_ended_at,
            duration_ms=duration_ms,
        )

    def _build_failed_response(
        self,
        payload: AnalyzeEvidenceRequest,
        processing_started_at: datetime,
        failure_reason: str,
        detected_signals: list[str] | None = None,
    ) -> AnalyzeEvidenceResponse:
        processing_finished_at = datetime.now(UTC)

        return AnalyzeEvidenceResponse(
            analysis_id=f"failed-analysis-{payload.evidence_record_id}",
            analysis_version=ANALYSIS_VERSION,
            status=AnalysisStatus.FAILED,
            risk_level=RiskLevel.UNKNOWN,
            confidence=0,
            summary="Audio transcription failed before risk classification.",
            detected_signals=[
                "analysis_failed",
                failure_reason,
                *(detected_signals or []),
            ],
            should_escalate=False,
            recommended_action=RecommendedAction.REVIEW,
            evidence_window=self._get_evidence_window(payload),
            transcription=None,
            acoustic_events=[],
            threat_matches=[],
            provider_metadata=AnalysisProviderMetadata(
                provider="unavailable",
                model="unavailable",
                model_version=ANALYSIS_VERSION,
            ),
            processing_started_at=processing_started_at,
            processing_finished_at=processing_finished_at,
            latency_ms=self._get_latency_ms(
                processing_started_at,
                processing_finished_at,
            ),
            failure_reason=failure_reason,
        )

    def _get_latency_ms(self, started_at: datetime, finished_at: datetime) -> int:
        return max(0, round((finished_at - started_at).total_seconds() * 1000))
