from datetime import UTC, datetime

from app.schemas.analyze import (
    AcousticEvent,
    AnalysisProviderMetadata,
    AnalysisStatus,
    AnalyzeEvidenceRequest,
    AnalyzeEvidenceResponse,
    EvidenceWindow,
    RecommendedAction,
    RiskLevel,
)

ANALYSIS_VERSION = "audio-evidence-v1"
MOCK_PROCESSING_STARTED_AT = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
MOCK_PROCESSING_FINISHED_AT = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)


class AnalyzeService:
    def analyze(self, payload: AnalyzeEvidenceRequest) -> AnalyzeEvidenceResponse:
        return AnalyzeEvidenceResponse(
            analysis_id=f"mock-analysis-{payload.evidence_record_id}",
            analysis_version=ANALYSIS_VERSION,
            status=AnalysisStatus.COMPLETED,
            risk_level=RiskLevel.LOW,
            confidence=0.12,
            summary=(
                "Mock analysis completed. No real model was executed and no "
                "critical escalation was inferred from metadata-only input."
            ),
            detected_signals=[
                "mock_analysis",
                "metadata_received",
                f"evidence_type:{payload.evidence_type}",
            ],
            should_escalate=False,
            recommended_action=RecommendedAction.NONE,
            evidence_window=self._get_evidence_window(payload),
            transcription=None,
            acoustic_events=[
                AcousticEvent(
                    label="mock_metadata_only_analysis",
                    start_ms=0,
                    end_ms=0,
                    confidence=0.12,
                    source="mock",
                )
            ],
            threat_matches=[],
            provider_metadata=AnalysisProviderMetadata(
                provider="mock",
                model="metadata-only",
                model_version=ANALYSIS_VERSION,
            ),
            processing_started_at=MOCK_PROCESSING_STARTED_AT,
            processing_finished_at=MOCK_PROCESSING_FINISHED_AT,
            latency_ms=1000,
            failure_reason=None,
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
