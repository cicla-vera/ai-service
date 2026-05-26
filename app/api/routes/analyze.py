from fastapi import APIRouter

from app.schemas.analyze import AnalyzeEvidenceRequest, AnalyzeEvidenceResponse
from app.services.analyze_service import AnalyzeService

router = APIRouter(tags=["analysis"])
analyze_service = AnalyzeService()


@router.post("/analyze", response_model=AnalyzeEvidenceResponse)
def analyze_evidence(payload: AnalyzeEvidenceRequest) -> AnalyzeEvidenceResponse:
    return analyze_service.analyze(payload)
