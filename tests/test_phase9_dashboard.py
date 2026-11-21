"""Automated tests for Phase 9: IndicRAG-QA Interactive Research Dashboard.

Validates:
1. API health, runtime metadata, and environment specifications.
2. Static Single-Page Application (SPA) serving and route handling.
3. Live query execution across systems S0 through S5.
4. Script-aware answer behavior (Roman Hinglish, Devanagari Hindi, English).
5. Granular LangGraph agent execution traces and telemetry.
6. Searchable Evidence Explorer filtering and pagination.
7. Benchmark metrics data consistency with frozen Phase 8 artifacts.
8. Strict isolation of the offline Judge Agent from the runtime serving pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indicrag.web.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a FastAPI test client for dashboard tests."""
    with TestClient(app) as c:
        yield c


def test_dashboard_health_endpoint(client: TestClient) -> None:
    """Verify health endpoint returns valid metadata, seed, and protocol."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["seed"] == 42
    assert "RGB Benchmark" in data["protocol"]
    assert "CPU-only" in data["environment"]
    assert "serving_answerer" in data["models"]
    assert "offline_judge" in data["models"]


def test_query_samples_availability(client: TestClient) -> None:
    """Verify pre-configured sample questions exist covering multi-script scenarios."""
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 4

    scripts = [s["script"] for s in samples]
    assert "roman_hinglish" in scripts
    assert "devanagari" in scripts
    assert "latin_english" in scripts


def test_metrics_overview_consistency(client: TestClient) -> None:
    """Verify metrics overview endpoint matches Phase 8 benchmark numbers."""
    res = client.get("/api/metrics/overview")
    assert res.status_code == 200
    data = res.json()

    assert data["benchmark_counts"]["dev_total"] == 150
    assert data["benchmark_counts"]["dev_answerable"] == 75
    assert data["benchmark_counts"]["dev_unanswerable"] == 75
    assert data["benchmark_counts"]["test_total"] == 75
    assert data["benchmark_counts"]["clean_corpus_passages"] == 2078
    assert data["benchmark_counts"]["perturbed_corpus_passages"] == 2493
    assert data["metric_identities_valid"] is True
    assert data["key_highlights"]["best_clean_acc_ans"] == 92.00
    assert data["key_highlights"]["unanswerable_rejection_rate"] == 100.00


def test_systems_ladder_endpoints(client: TestClient) -> None:
    """Verify systems ladder endpoints for Dev and Test splits."""
    # Dev
    res_dev = client.get("/api/metrics/ladder?split=dev")
    assert res_dev.status_code == 200
    rows_dev = res_dev.json()
    assert len(rows_dev) == 12  # S0-S5 x clean/perturbed
    systems_dev = {r["system"] for r in rows_dev}
    assert {"S0", "S1", "S2", "S3", "S4", "S5"} == systems_dev

    # Test
    res_test = client.get("/api/metrics/ladder?split=test")
    assert res_test.status_code == 200
    rows_test = res_test.json()
    assert len(rows_test) == 12


def test_noise_and_ablation_endpoints(client: TestClient) -> None:
    """Verify noise degradation, Hinglish ablation, and retry matrix endpoints."""
    # Noise
    res_noise = client.get("/api/metrics/noise")
    assert res_noise.status_code == 200
    assert len(res_noise.json()) == 12

    # Hinglish
    res_h = client.get("/api/metrics/hinglish")
    assert res_h.status_code == 200
    h_rows = res_h.json()
    assert len(h_rows) == 3
    methods = [r["method"] for r in h_rows]
    assert "Method A: Unnormalized Roman" in methods

    # Retry Matrix
    res_retry = client.get("/api/metrics/retry")
    assert res_retry.status_code == 200
    retry_rows = res_retry.json()
    total_count = sum(r["count"] for r in retry_rows)
    assert total_count == 75


def test_evidence_explorer_search(client: TestClient) -> None:
    """Verify evidence explorer search, filtering, and pagination."""
    # Blank query
    res = client.get("/api/evidence/passages?corpus=clean&limit=10&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2078
    assert len(data["passages"]) == 10

    # Keyword query
    res_kw = client.get("/api/evidence/passages?query=delhi&corpus=clean&limit=5")
    assert res_kw.status_code == 200
    assert len(res_kw.json()["passages"]) <= 5

    # Perturbed corpus check
    res_pert = client.get("/api/evidence/passages?corpus=perturbed&limit=10")
    assert res_pert.status_code == 200
    assert res_pert.json()["total"] == 2493


def test_live_query_execution_hinglish(client: TestClient) -> None:
    """Verify live query execution with Roman Hinglish input."""
    payload = {
        "question": "Laxmi Mittal ka business kisme hai?",
        "system": "S5",
        "corpus": "clean",
    }
    res = client.post("/api/query/execute", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["detected_script"] == "roman_hinglish"
    assert data["detected_language"] == "hi-Latn"
    assert data["system_ladder"] == "S5"
    assert data["corpus_condition"] == "clean"
    assert len(data["retrieved_passages"]) > 0
    assert len(data["traces"]) >= 5
    assert "stats" in data
    assert data["stats"]["latency_ms"] > 0


def test_live_query_execution_devanagari(client: TestClient) -> None:
    """Verify live query execution with Devanagari Hindi input."""
    payload = {
        "question": "लक्ष्मी निवास मित्तल का जन्म किस राज्य में हुआ था?",
        "system": "S5",
        "corpus": "clean",
    }
    res = client.post("/api/query/execute", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["detected_script"] == "devanagari"
    assert data["detected_language"] == "hi"
    assert len(data["retrieved_passages"]) > 0
    assert len(data["traces"]) >= 5


def test_judge_agent_isolation_in_dashboard(client: TestClient) -> None:
    """Verify that the offline JudgeAgent is NEVER invoked during serving execution."""
    # Execute a live query
    res = client.post("/api/query/execute", json={"question": "Test query", "system": "S5"})
    assert res.status_code == 200
    traces = res.json()["traces"]

    # Ensure no node in the graph executed a judge call
    node_names = [t["node"] for t in traces]
    assert "judge_agent" not in node_names
    assert "offline_judge" not in node_names
    assert "judge" not in node_names


def test_spa_static_files_served(client: TestClient) -> None:
    """Verify SPA index.html and static assets are accessible."""
    # Index HTML
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "IndicRAG-QA" in res_index.text
    assert "Hinglish QA Workspace" in res_index.text

    # CSS
    res_css = client.get("/static/css/style.css")
    assert res_css.status_code == 200
    assert "--bg-primary" in res_css.text

    # JS
    res_js = client.get("/static/js/app.js")
    assert res_js.status_code == 200
    assert "detectScriptClient" in res_js.text
