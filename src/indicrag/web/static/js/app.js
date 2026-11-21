/**
 * IndicRAG-QA Main Application Controller.
 * Modular, reactive SPA orchestrating navigation, query execution, graph tracing,
 * evidence exploration, and benchmark evaluation lab.
 */

import { Api } from "./api.js";
import { Charts } from "./charts.js";

// Global State
const State = {
  activeView: "overview",
  currentSplit: "dev",
  samples: [],
  overviewData: null,
  ladderData: null,
  lastQueryResult: null,
  isExecuting: false,
};

// Distinctive Hinglish and English lexical markers for real-time frontend detection
const HINGLISH_MARKERS = new Set([
  "kya", "kyu", "kyun", "kaun", "kahan", "kab", "kaise", "kitna", "kitne", "kitni",
  "hai", "hain", "tha", "thi", "hoga", "hogi", "mein", "ko", "ka", "ki", "ke", "ne",
  "par", "pe", "aur", "bhi", "nahi", "nahin", "yeh", "woh", "unka", "batao", "karo"
]);

const ENGLISH_WORDS = new Set([
  "what", "is", "the", "capital", "of", "in", "to", "for", "with", "on", "at",
  "from", "by", "about", "who", "which", "where", "when", "why", "how", "are", "was"
]);

function detectScriptClient(text) {
  if (!text || !text.trim()) return { script: "unknown", language: "unknown", label: "Ready" };

  const devanagariCount = (text.match(/[\u0900-\u097F]/g) || []).length;
  const latinCount = (text.match(/[a-zA-Z]/g) || []).length;
  const total = devanagariCount + latinCount;

  if (total === 0) return { script: "unknown", language: "unknown", label: "Ready" };
  if (devanagariCount / total >= 0.35) {
    return { script: "devanagari", language: "hindi", label: "Hindi (Devanagari)" };
  }

  const tokens = (text.toLowerCase().match(/\b[a-zA-Z]+\b/g) || []);
  let hinglishHits = 0;
  let englishHits = 0;

  tokens.forEach(t => {
    if (HINGLISH_MARKERS.has(t)) hinglishHits++;
    if (ENGLISH_WORDS.has(t)) englishHits++;
  });

  if (hinglishHits > englishHits || (hinglishHits > 0 && englishHits === 0)) {
    return { script: "roman_hinglish", language: "hinglish", label: "Hinglish (Roman Script)" };
  }

  return { script: "latin_english", language: "english", label: "English (Latin Script)" };
}

// ---------------------------------------------------------------------------
// Navigation Controller
// ---------------------------------------------------------------------------
function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", e => {
      e.preventDefault();
      const targetView = item.dataset.view;
      if (targetView) switchView(targetView);
    });
  });

  // Topbar quick sample dropdown
  const sampleSelect = document.getElementById("topbar-sample-select");
  if (sampleSelect) {
    sampleSelect.addEventListener("change", e => {
      const sampleId = e.target.value;
      if (!sampleId) return;
      loadSampleIntoWorkspace(sampleId);
      switchView("workspace");
      e.target.value = "";
    });
  }
}

function switchView(viewName) {
  State.activeView = viewName;

  // Update nav items
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.view === viewName);
  });

  // Update views
  document.querySelectorAll(".page-view").forEach(el => {
    el.classList.toggle("active", el.id === `view-${viewName}`);
  });

  // Update topbar breadcrumb
  const breadcrumbEl = document.getElementById("active-breadcrumb");
  if (breadcrumbEl) {
    const titles = {
      overview: "Research Operations Overview",
      workspace: "Hinglish QA Workspace",
      agents: "Agent Operations & Graph Telemetry",
      evidence: "Evidence & Corpus Explorer",
      eval: "Evaluation Lab & RGB Benchmark",
    };
    breadcrumbEl.innerHTML = `IndicRAG-QA <span>/ ${titles[viewName] || viewName}</span>`;
  }

  // View-specific data triggers
  if (viewName === "overview" && !State.overviewData) loadOverviewData();
  if (viewName === "eval") loadEvaluationLabData();
  if (viewName === "evidence") loadEvidenceExplorerData();
}

// ---------------------------------------------------------------------------
// Overview View
// ---------------------------------------------------------------------------
async function loadOverviewData() {
  try {
    const overview = await Api.getOverview();
    State.overviewData = overview;

    // Render Metric Cards
    const elClean = document.getElementById("metric-clean-acc");
    if (elClean) elClean.innerText = `${overview.key_highlights.best_clean_acc_ans.toFixed(1)}%`;

    const elFrr = document.getElementById("metric-frr");
    if (elFrr) elFrr.innerText = `${overview.key_highlights.unnormalized_hinglish_acc_ans.toFixed(2)}%`;

    const elUnans = document.getElementById("metric-unans-acc");
    if (elUnans) elUnans.innerText = `${overview.key_highlights.unanswerable_rejection_rate.toFixed(1)}%`;

    const elDrop = document.getElementById("metric-noise-drop");
    if (elDrop) elDrop.innerText = `${overview.key_highlights.clean_to_perturbed_drop_pp.toFixed(1)} pp`;

    // Render Ladder comparison chart on overview
    const ladder = await Api.getSystemsLadder("dev");
    State.ladderData = ladder;
    Charts.renderSystemsLadder("overview-ladder-chart", ladder);

    // Render Hinglish ablation chart
    const hinglish = await Api.getHinglishAblation();
    Charts.renderHinglishAblation("overview-hinglish-chart", hinglish);
  } catch (err) {
    console.error("Failed to load overview data:", err);
  }
}

// ---------------------------------------------------------------------------
// Hinglish QA Workspace View
// ---------------------------------------------------------------------------
function initWorkspace() {
  const textarea = document.getElementById("query-input");
  const scriptChip = document.getElementById("detected-script-chip");
  const runBtn = document.getElementById("run-query-btn");
  const clearBtn = document.getElementById("clear-query-btn");

  if (textarea && scriptChip) {
    textarea.addEventListener("input", () => {
      const text = textarea.value;
      const detected = detectScriptClient(text);
      scriptChip.innerText = detected.label;
      scriptChip.className = "chip chip-cyan";
    });

    textarea.addEventListener("keydown", e => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        triggerQueryExecution();
      }
    });
  }

  if (runBtn) {
    runBtn.addEventListener("click", triggerQueryExecution);
  }

  if (clearBtn && textarea) {
    clearBtn.addEventListener("click", () => {
      textarea.value = "";
      textarea.focus();
      scriptChip.innerText = "Ready";
    });
  }

  // Pre-load sample buttons
  renderWorkspaceSamples();
}

async function renderWorkspaceSamples() {
  try {
    const samples = await Api.getSamples();
    State.samples = samples;

    const container = document.getElementById("sample-buttons-container");
    const topbarSelect = document.getElementById("topbar-sample-select");

    if (container) {
      container.innerHTML = "";
      samples.forEach(s => {
        const btn = document.createElement("button");
        btn.className = "sample-btn";
        btn.innerText = s.label;
        btn.title = s.expected_behavior;
        btn.addEventListener("click", () => loadSampleIntoWorkspace(s.id));
        container.appendChild(btn);
      });
    }

    if (topbarSelect) {
      topbarSelect.innerHTML = '<option value="">Load Sample Query...</option>';
      samples.forEach(s => {
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.innerText = s.label;
        topbarSelect.appendChild(opt);
      });
    }
  } catch (err) {
    console.warn("Could not load samples:", err);
  }
}

function loadSampleIntoWorkspace(sampleId) {
  const sample = State.samples.find(s => s.id === sampleId);
  if (!sample) return;

  const textarea = document.getElementById("query-input");
  const sysSelect = document.getElementById("system-select");
  const scriptChip = document.getElementById("detected-script-chip");

  if (textarea) {
    textarea.value = sample.question;
    textarea.focus();
  }
  if (sysSelect) sysSelect.value = sample.recommended_system || "S5";
  if (scriptChip) {
    const detected = detectScriptClient(sample.question);
    scriptChip.innerText = detected.label;
  }
}

async function triggerQueryExecution() {
  const textarea = document.getElementById("query-input");
  const sysSelect = document.getElementById("system-select");
  const corpusSelect = document.getElementById("corpus-select");
  const runBtn = document.getElementById("run-query-btn");

  if (!textarea || State.isExecuting) return;
  const question = textarea.value.trim();
  if (!question) {
    alert("Please enter a question to execute.");
    return;
  }

  const system = sysSelect ? sysSelect.value : "S5";
  const corpus = corpusSelect ? corpusSelect.value : "clean";

  // Set loading state
  State.isExecuting = true;
  runBtn.disabled = true;
  runBtn.innerHTML = `
    <svg class="spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;">
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
    Executing Multi-Agent Graph...
  `;

  const ansBox = document.getElementById("result-answer");
  const passagesBox = document.getElementById("retrieved-passages-list");
  const statusBadges = document.getElementById("result-badges");

  if (ansBox) ansBox.innerHTML = '<span style="color:var(--text-muted); font-style:italic;">Routing through supervisor, retriever, evidence grader, answerer, and verifier...</span>';
  if (passagesBox) passagesBox.innerHTML = "";

  try {
    const startTime = performance.now();
    const result = await Api.executeQuery(question, system, corpus);
    const duration = (performance.now() - startTime).toFixed(0);

    State.lastQueryResult = result;

    // Render result
    renderQueryResult(result, duration);

    // Update Agent Operations view with latest trace
    renderAgentOperations(result);
  } catch (err) {
    console.error("Query execution error:", err);
    if (ansBox) {
      ansBox.innerHTML = `<span style="color:var(--rose-primary); font-weight:600;">Execution Error: ${err.message}</span>`;
    }
  } finally {
    State.isExecuting = false;
    runBtn.disabled = false;
    runBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polygon points="5 3 19 12 5 21 5 3" />
      </svg>
      Run Query (Ctrl + Enter)
    `;
  }
}

function renderQueryResult(result, durationMs) {
  const ansBox = document.getElementById("result-answer");
  const abstentionBanner = document.getElementById("abstention-banner");
  const badgesBox = document.getElementById("result-badges");
  const passagesList = document.getElementById("retrieved-passages-list");
  const statsBox = document.getElementById("result-stats");

  // Format citations into clickable chips [psg_xxx]
  let formattedAnswer = escapeHtml(result.final_answer);
  formattedAnswer = formattedAnswer.replace(/\[([a-zA-Z0-9_\-\.]+)\]/g, (match, pid) => {
    return `<a href="#passage-${pid}" class="citation-chip" title="Inspect source passage">[${pid}]</a>`;
  });

  if (ansBox) ansBox.innerHTML = formattedAnswer;

  // Abstention alert
  if (abstentionBanner) {
    if (result.is_abstention) {
      abstentionBanner.style.display = "flex";
      abstentionBanner.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        <span><strong>Abstention Enforced:</strong> Reason Code <code>${result.reason_code || 'UNSPECIFIED'}</code> — System safely abstained without hallucinating evidence.</span>
      `;
    } else {
      abstentionBanner.style.display = "none";
    }
  }

  // Status Badges
  if (badgesBox) {
    badgesBox.innerHTML = `
      <span class="chip chip-cyan">${result.system_ladder}</span>
      <span class="chip chip-emerald">${result.detected_script.toUpperCase()}</span>
      <span class="chip">${result.corpus_condition.toUpperCase()} CORPUS</span>
      <span class="chip">${durationMs} ms</span>
    `;
  }

  // Execution Telemetry Stats
  if (statsBox) {
    const s = result.stats;
    statsBox.innerHTML = `
      <div class="chip">Rounds: ${s.retrieval_rounds}</div>
      <div class="chip">Retries: ${s.retry_count}</div>
      <div class="chip">LLM Calls: ${s.llm_calls}</div>
      <div class="chip">Latency: ${s.latency_ms} ms</div>
    `;
  }

  // Render Retrieved Passages
  if (passagesList) {
    if (!result.retrieved_passages || result.retrieved_passages.length === 0) {
      passagesList.innerHTML = '<div style="color:var(--text-muted); padding:10px;">Zero passages retrieved for this query.</div>';
    } else {
      passagesList.innerHTML = result.retrieved_passages.map((p, idx) => `
        <div class="passage-card" id="passage-${p.passage_id}">
          <div class="passage-header">
            <span class="passage-id">[${p.passage_id}]</span>
            <div>
              ${p.is_counterfactual ? '<span class="chip chip-amber" style="font-size:10px;">Counterfactual Distractor</span>' : ''}
              <span class="chip chip-cyan" style="font-size:10px;">Score: ${p.score.toFixed(4)}</span>
            </div>
          </div>
          <div class="passage-title">${escapeHtml(p.title || 'Untitled Passage')}</div>
          <div class="passage-text">${escapeHtml(p.text)}</div>
        </div>
      `).join("");
    }
  }
}

// ---------------------------------------------------------------------------
// Agent Operations View (Graph Flow & Step Traces)
// ---------------------------------------------------------------------------
function renderAgentOperations(result) {
  if (!result || !result.traces) return;

  const nodeMap = {
    supervisor: document.getElementById("node-supervisor"),
    script_detector: document.getElementById("node-script"),
    query_agent: document.getElementById("node-query"),
    retriever_agent: document.getElementById("node-retriever"),
    evidence_grader: document.getElementById("node-grader"),
    answerer_agent: document.getElementById("node-answerer"),
    verifier_agent: document.getElementById("node-verifier"),
    abstention_arbiter: document.getElementById("node-arbiter"),
  };

  // Reset node styles
  Object.values(nodeMap).forEach(el => {
    if (el) el.className = "node-box";
  });

  // Mark executed nodes
  result.traces.forEach(t => {
    const el = nodeMap[t.node];
    if (el) {
      el.classList.add("active");
      if (t.action.includes("retry") || t.action.includes("trigger_retry")) {
        el.classList.add("retrying");
      } else if (t.is_abstention) {
        el.classList.add("abstention");
      } else {
        el.classList.add("success");
      }
    }
  });

  // Render Trace Timeline
  const timelineEl = document.getElementById("trace-timeline-container");
  if (timelineEl) {
    timelineEl.innerHTML = result.traces.map((t, i) => `
      <div class="trace-item">
        <span class="trace-index">#${i + 1}</span>
        <span class="trace-node">${t.node}</span>
        <span class="trace-action">${t.action}</span>
        <span class="trace-details">Round ${t.round} | Retry ${t.retry_count} | Calls: ${t.llm_calls} ${t.reason_code ? `| Reason: ${t.reason_code}` : ''}</span>
      </div>
    `).join("");
  }
}

// ---------------------------------------------------------------------------
// Evidence Explorer View
// ---------------------------------------------------------------------------
async function loadEvidenceExplorerData() {
  const searchInput = document.getElementById("evidence-search-input");
  const corpusSelect = document.getElementById("evidence-corpus-select");
  const container = document.getElementById("evidence-table-body");
  const totalEl = document.getElementById("evidence-total-count");

  const query = searchInput ? searchInput.value.trim() : "";
  const corpus = corpusSelect ? corpusSelect.value : "clean";

  if (container) container.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">Searching corpus index...</td></tr>';

  try {
    const data = await Api.searchPassages(query, corpus, 25, 0);
    if (totalEl) totalEl.innerText = `${data.total.toLocaleString()} Passages Indexed`;

    if (container) {
      if (data.passages.length === 0) {
        container.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">No passages matching query criteria.</td></tr>';
        return;
      }

      container.innerHTML = data.passages.map(p => `
        <tr>
          <td><code class="citation-chip">[${p.passage_id}]</code></td>
          <td style="font-weight:600; color:#fff;">${escapeHtml(p.title)}</td>
          <td><span class="chip ${p.is_counterfactual ? 'chip-amber' : 'chip-cyan'}">${p.is_counterfactual ? 'Counterfactual Distractor' : p.source_partition}</span></td>
          <td style="font-family:var(--font-mono);">${p.char_length} chars</td>
          <td style="max-width:320px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text-secondary);">${escapeHtml(p.text)}</td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Evidence explorer load failure:", err);
    if (container) container.innerHTML = `<tr><td colspan="5" style="color:var(--rose-primary);">Error loading passages: ${err.message}</td></tr>`;
  }
}

// ---------------------------------------------------------------------------
// Evaluation Lab View
// ---------------------------------------------------------------------------
async function loadEvaluationLabData() {
  try {
    // 1. Systems Ladder Table
    const ladderData = await Api.getSystemsLadder("dev");
    const ladderBody = document.getElementById("ladder-table-body");
    if (ladderBody && ladderData) {
      ladderBody.innerHTML = ladderData.map(r => `
        <tr class="${r.system === 'S5' ? 'highlight' : ''}">
          <td style="font-family:var(--font-mono); font-weight:700; color:var(--cyan-primary);">${r.system}</td>
          <td><span class="chip ${r.corpus === 'perturbed' ? 'chip-rose' : 'chip-cyan'}">${(r.corpus || 'clean').toUpperCase()}</span></td>
          <td>${r.acc_ans.toFixed(2)}%</td>
          <td>${(r.false_refusal_rate || r.frr).toFixed(2)}%</td>
          <td>${(r.error_rate_ans || r.err_ans || 0).toFixed(2)}%</td>
          <td>${r.acc_unans.toFixed(2)}%</td>
          <td>${(r.hallucination_rate || r.hr || 0).toFixed(2)}%</td>
          <td>${r.acc_overall.toFixed(2)}%</td>
        </tr>
      `).join("");
    }

    // 2. Retry Transition Matrix (75 queries)
    const retryData = await Api.getRetryMatrix();
    const retryBody = document.getElementById("retry-table-body");
    if (retryBody && retryData) {
      retryBody.innerHTML = retryData.map(r => `
        <tr>
          <td style="font-family:var(--font-mono); font-weight:600;">${r.initial_s2_state}</td>
          <td style="font-family:var(--font-mono); font-weight:600;">${r.s5_outcome}</td>
          <td><span class="chip ${r.count === 5 ? 'chip-amber' : (r.count === 63 ? 'chip-emerald' : 'chip-rose')}">${r.transition_type}</span></td>
          <td style="font-weight:700;">${r.count}</td>
          <td>${r.pct.toFixed(2)}%</td>
        </tr>
      `).join("");
    }

    // 3. Statistical Significance
    const sigData = await Api.getSignificance();
    const sigBody = document.getElementById("sig-table-body");
    if (sigBody && sigData) {
      sigBody.innerHTML = sigData.map(r => `
        <tr>
          <td style="font-family:var(--font-mono); font-weight:600;">${r.comparison}</td>
          <td>${r.contingency}</td>
          <td>${r.discordant_pairs}</td>
          <td style="font-weight:700;">${r.p_value_raw.toFixed(4)}</td>
          <td><span class="chip ${r.bonferroni_sig ? 'chip-emerald' : 'chip-rose'}">${r.bonferroni_sig ? 'SIGNIFICANT' : 'NOT SIGNIFICANT (p ≥ 0.05)'}</span></td>
          <td style="font-size:12px; color:var(--text-secondary);">${r.interpretation}</td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Evaluation lab data load error:", err);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Initialize on DOM load
window.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initWorkspace();

  // Search input debouncer in Evidence Explorer
  const evidenceSearch = document.getElementById("evidence-search-input");
  const evidenceCorpus = document.getElementById("evidence-corpus-select");
  if (evidenceSearch) {
    let timeout = null;
    evidenceSearch.addEventListener("input", () => {
      clearTimeout(timeout);
      timeout = setTimeout(loadEvidenceExplorerData, 300);
    });
  }
  if (evidenceCorpus) {
    evidenceCorpus.addEventListener("change", loadEvidenceExplorerData);
  }

  // Load initial view
  loadOverviewData();
});
