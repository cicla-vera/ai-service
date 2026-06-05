from __future__ import annotations

import os

from app.evaluation import render_markdown_report, run_evaluation


def test_evaluation_harness_passes_mock_cases(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "deepgram")
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER_CHAIN", "deepgram,groq")

    report = run_evaluation(provider_mode="mock", cost_usd_per_hour=1.5)

    assert report["caseCount"] == 6
    assert report["passed"] == 6
    assert report["failed"] == 0
    assert report["falsePositiveCount"] == 0
    assert report["falseNegativeCount"] == 0
    assert report["estimatedCostUsd"] == 0.003
    assert report["thresholds"]["acoustic"] == {
        "windowMs": 100,
        "highRmsThreshold": 0.42,
        "peakThreshold": 0.82,
        "impactDeltaThreshold": 0.35,
        "clippingRatioThreshold": 0.03,
    }
    assert "concrete lethal threat text" in report["thresholds"]["riskPolicy"][
        "criticalRules"
    ]
    assert {
        item["id"]: item["actualRisk"]
        for item in report["cases"]
    } == {
        "benign_quiet": "LOW",
        "tense_loud_no_threat": "MEDIUM",
        "severe_verbal_abuse": "MEDIUM",
        "physical_assault_threat": "HIGH",
        "lethal_threat": "CRITICAL",
        "distress_with_impact": "CRITICAL",
    }
    assert {
        item["id"]: item["actualShouldEscalate"]
        for item in report["cases"]
    } == {
        "benign_quiet": False,
        "tense_loud_no_threat": False,
        "severe_verbal_abuse": False,
        "physical_assault_threat": False,
        "lethal_threat": True,
        "distress_with_impact": True,
    }
    assert "consented local fixtures" in "\n".join(report["limitations"])


def test_evaluation_harness_restores_provider_env(monkeypatch) -> None:
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER", "openai")
    monkeypatch.setenv("AI_TRANSCRIPTION_PROVIDER_CHAIN", "openai")
    monkeypatch.setenv("AI_MOCK_TRANSCRIPTION_TEXT", "existing")

    run_evaluation(provider_mode="mock")

    assert os.environ["AI_TRANSCRIPTION_PROVIDER"] == "openai"
    assert os.environ["AI_TRANSCRIPTION_PROVIDER_CHAIN"] == "openai"
    assert os.environ["AI_MOCK_TRANSCRIPTION_TEXT"] == "existing"


def test_evaluation_markdown_report_includes_summary_and_limitations() -> None:
    report = run_evaluation(provider_mode="mock")

    markdown = render_markdown_report(report)

    assert "# Vera AI Audio Evaluation" in markdown
    assert "- Cases: `6/6` passed" in markdown
    assert "- False positives: `0`" in markdown
    assert "- False negatives: `0`" in markdown
    assert "## Thresholds" in markdown
    assert "- Acoustic high RMS threshold: `0.42`" in markdown
    assert "## Limitations" in markdown
    assert "synthetic" in markdown
