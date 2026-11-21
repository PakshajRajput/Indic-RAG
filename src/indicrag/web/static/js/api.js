/**
 * IndicRAG-QA Web API Service Client.
 * Handles communication with the FastAPI backend.
 */

const API_BASE = window.location.origin;

export const Api = {
  /**
   * Health and runtime metadata check.
   */
  async getHealth() {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch pre-configured sample questions.
   */
  async getSamples() {
    const res = await fetch(`${API_BASE}/api/samples`);
    if (!res.ok) throw new Error(`Failed to load samples: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch executive benchmark metrics overview.
   */
  async getOverview() {
    const res = await fetch(`${API_BASE}/api/metrics/overview`);
    if (!res.ok) throw new Error(`Failed to load metrics overview: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch S0-S5 Systems Ladder metrics.
   */
  async getSystemsLadder(split = "dev") {
    const res = await fetch(`${API_BASE}/api/metrics/ladder?split=${encodeURIComponent(split)}`);
    if (!res.ok) throw new Error(`Failed to load systems ladder: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch Clean vs. Perturbed noise degradation table.
   */
  async getNoiseDegradation() {
    const res = await fetch(`${API_BASE}/api/metrics/noise`);
    if (!res.ok) throw new Error(`Failed to load noise degradation: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch Cross-Script Hinglish ablation table.
   */
  async getHinglishAblation() {
    const res = await fetch(`${API_BASE}/api/metrics/hinglish`);
    if (!res.ok) throw new Error(`Failed to load Hinglish ablation: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch Retry transition matrix (75 answerable queries).
   */
  async getRetryMatrix() {
    const res = await fetch(`${API_BASE}/api/metrics/retry`);
    if (!res.ok) throw new Error(`Failed to load retry matrix: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Fetch Paired statistical significance tests.
   */
  async getSignificance() {
    const res = await fetch(`${API_BASE}/api/metrics/significance`);
    if (!res.ok) throw new Error(`Failed to load significance report: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Search and filter passages in the corpus.
   */
  async searchPassages(query = "", corpus = "clean", limit = 20, offset = 0) {
    const url = new URL(`${API_BASE}/api/evidence/passages`);
    url.searchParams.set("query", query);
    url.searchParams.set("corpus", corpus);
    url.searchParams.set("limit", limit.toString());
    url.searchParams.set("offset", offset.toString());

    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`Failed to search passages: ${res.statusText}`);
    return await res.json();
  },

  /**
   * Execute live multi-agent query through LangGraph.
   */
  async executeQuery(question, system = "S5", corpus = "clean") {
    const res = await fetch(`${API_BASE}/api/query/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, system, corpus }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Query execution failed");
    }

    return await res.json();
  },
};
