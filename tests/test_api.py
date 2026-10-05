"""HTTP contract tests for health, query, and voice evaluation."""

from __future__ import annotations

from tests.conftest import AUTH_HEADERS


def test_health_reports_injected_services(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chroma_ready"] is True
    assert body["asr_ready"] is True
    assert body["llm_configured"] is True


def test_query_requires_bearer_token(client) -> None:
    response = client.post("/api/v1/query", json={"query_text": "rolling update"})
    assert response.status_code == 401


def test_query_returns_rag_chunks(client) -> None:
    response = client.post(
        "/api/v1/query",
        headers=AUTH_HEADERS,
        json={"query_text": "Kubernetes rolling updates", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["chunks"][0]["source"] == "kubernetes/docs/deployments.md"


def test_evaluate_voice_pipeline(client) -> None:
    response = client.post(
        "/api/v1/evaluate/voice",
        headers=AUTH_HEADERS,
        files={"audio_file": ("answer.webm", b"fake-webm-bytes", "audio/webm")},
        data={"target_scenario": "system_design_interview"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert "execution_time_ms" in body
    assert body["asr_metrics"]["transcription"]
    assert 0 <= body["evaluation_core"]["overall_score"] <= 100
    assert body["feedback_details"][0]["rag_source_context"]


def test_evaluate_rejects_empty_audio(client) -> None:
    response = client.post(
        "/api/v1/evaluate/voice",
        headers=AUTH_HEADERS,
        files={"audio_file": ("answer.webm", b"", "audio/webm")},
    )
    assert response.status_code == 400
