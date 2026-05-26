from app.schemas.analyze import (
    AnalyzeEvidenceRequest,
    AnalyzeEvidenceResponse,
    RiskLevel,
)


class AnalyzeService:
    def analyze(self, payload: AnalyzeEvidenceRequest) -> AnalyzeEvidenceResponse:
        return AnalyzeEvidenceResponse(
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
        )
