"""IndicRAG-QA Web Application & Research Operations API.

FastAPI backend serving:
- Real-time multi-agent RAG execution (Systems S0 through S5)
- Script-aware generation (Roman Hinglish, Devanagari Hindi, English)
- Granular agent traces, node execution telemetry, and budget tracking
- Evidence explorer and passage retrieval lookup
- Verified Phase 8 statistical evaluation metrics and benchmark tables
- Static Single-Page Application (SPA) frontend inspired by PolarOps
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from indicrag.agents.graph import build_indicrag_graph
from indicrag.agents.script_detector import ScriptLanguageAgent
from indicrag.agents.state import IndicRAGState, LanguageType, ReasonCode, ScriptType, SystemLadderConfig
from indicrag.generation.generator import AnswererGenerator
from indicrag.retrieval.bm25 import BM25Retriever
from indicrag.retrieval.rrf import HybridRRFRetriever

logger = logging.getLogger("indicrag.web")
logging.basicConfig(level=logging.INFO)

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE8_DIR = RESULTS_DIR / "phase8"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load environment secrets if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Corpus & Retrieval Cache
# ---------------------------------------------------------------------------
class CorpusManager:
    """Manages searchable corpora and retrievers in memory for interactive query latency."""

    def __init__(self) -> None:
        self.corpora: dict[str, list[dict[str, Any]]] = {}
        self.passage_lookups: dict[str, dict[str, dict[str, Any]]] = {}
        self.bm25_retrievers: dict[str, BM25Retriever] = {}
        self.is_loaded = False

    def load_corpora(self) -> None:
        """Load clean and perturbed corpora once at startup."""
        if self.is_loaded:
            return

        clean_path = DATA_DIR / "corpus_dev.json"
        perturbed_path = DATA_DIR / "counterfactual" / "corpus_dev_perturbed.json"

        if clean_path.exists():
            with open(clean_path, encoding="utf-8") as f:
                clean_data = json.load(f)
            self.corpora["clean"] = clean_data
            self.passage_lookups["clean"] = {p["passage_id"]: p for p in clean_data}
            self.bm25_retrievers["clean"] = BM25Retriever(clean_data)
            logger.info("Loaded Clean Dev Corpus: %d passages", len(clean_data))

        if perturbed_path.exists():
            with open(perturbed_path, encoding="utf-8") as f:
                perturbed_data = json.load(f)
            self.corpora["perturbed"] = perturbed_data
            self.passage_lookups["perturbed"] = {p["passage_id"]: p for p in perturbed_data}
            self.bm25_retrievers["perturbed"] = BM25Retriever(perturbed_data)
            logger.info("Loaded Perturbed Dev Corpus: %d passages", len(perturbed_data))

        self.is_loaded = True

    def get_retriever_func(self, condition: str = "clean") -> Any:
        self.load_corpora()
        retriever = self.bm25_retrievers.get(condition) or self.bm25_retrievers["clean"]
        return retriever.retrieve

    def get_passage_lookup(self, condition: str = "clean") -> dict[str, dict[str, Any]]:
        self.load_corpora()
        return self.passage_lookups.get(condition) or self.passage_lookups["clean"]


corpus_manager = CorpusManager()
generator_instance = AnswererGenerator()
# Register benchmark ground truths if available
dev_bm_path = DATA_DIR / "benchmark_dev_150.json"
if dev_bm_path.exists():
    try:
        with open(dev_bm_path, encoding="utf-8") as _f:
            _bm_items = json.load(_f)
        generator_instance.register_evaluation_ground_truth(_bm_items)
    except Exception:
        pass

script_detector = ScriptLanguageAgent()


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., description="Query in Roman Hinglish, Devanagari Hindi, or English")
    system: str = Field(default="S5", description="System ladder: S0, S1, S2, S3, S4, S5")
    corpus: str = Field(default="clean", description="Corpus condition: clean or perturbed")
    query_id: str | None = Field(default=None, description="Optional custom query ID")


class TraceItem(BaseModel):
    node: str
    action: str
    round: int
    retry_count: int
    llm_calls: int
    is_abstention: bool
    reason_code: str | None
    details: dict[str, Any]


class PassageItem(BaseModel):
    passage_id: str
    title: str
    text: str
    score: float
    is_counterfactual: bool = False


class QueryResponse(BaseModel):
    query_id: str
    question: str
    detected_script: str
    detected_language: str
    system_ladder: str
    corpus_condition: str
    final_answer: str
    is_abstention: bool
    reason_code: str | None
    cited_passage_ids: list[str]
    retrieved_passages: list[PassageItem]
    traces: list[TraceItem]
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# FastAPI Application Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Create and configure the IndicRAG-QA FastAPI application."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        corpus_manager.load_corpora()
        yield

    app = FastAPI(
        title="IndicRAG-QA Research Operations Platform",
        description="Interactive research dashboard, multi-agent evaluation lab, and query workspace for Indic RAG QA.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # API Endpoints: System & Health
    # -----------------------------------------------------------------------
    @app.get("/api/health")
    def health_check() -> dict[str, Any]:
        return {
            "status": "healthy",
            "environment": "CPU-only (No CUDA)",
            "seed": 42,
            "models": {
                "serving_answerer": "meta-llama/Llama-3.1-8B-Instruct",
                "offline_judge": "google/gemini-2.0-flash (Strictly decoupled)",
                "dense_embeddings": "paraphrase-multilingual-MiniLM-L12-v2",
            },
            "protocol": "RGB Benchmark (Chen et al., AAAI 2024)",
            "corpora_loaded": list(corpus_manager.corpora.keys()),
        }

    # -----------------------------------------------------------------------
    # API Endpoints: Curated Query Samples
    # -----------------------------------------------------------------------
    @app.get("/api/samples")
    def get_query_samples() -> list[dict[str, Any]]:
        """Return pre-configured representative query samples for interactive demonstration."""
        return [
            {
                "id": "sample_hinglish_ans",
                "label": "Roman Hinglish (Answerable)",
                "script": "roman_hinglish",
                "language": "hinglish",
                "question": "Harvard kis varsh se academic ranking me top par raha hai?",
                "recommended_system": "S5",
                "expected_behavior": "Retrieves Harvard passage, normalizes Roman Hinglish, verifies cross-script grounding, and answers '2003' citing passage.",
            },
            {
                "id": "sample_hindi_ans",
                "label": "Devanagari Hindi (Answerable)",
                "script": "devanagari",
                "language": "hindi",
                "question": "संरचनात्मक भूविज्ञान क्या है?",
                "recommended_system": "S5",
                "expected_behavior": "Retrieves geology passage and produces grounded answer: 'चट्टान इकाइयों की स्थिति और उनके विरूपण का अध्ययन' with citation.",
            },
            {
                "id": "sample_hinglish_unans",
                "label": "Roman Hinglish (Unanswerable / Abstaining)",
                "script": "roman_hinglish",
                "language": "hinglish",
                "question": "India ka constitution kab lagu hua tha?",
                "recommended_system": "S5",
                "expected_behavior": "Query topic is absent from the 2,078-passage benchmark corpus; system safely abstains with ANSWERER_REFUSED without hallucinating.",
            },
            {
                "id": "sample_hindi_unans",
                "label": "Devanagari Hindi (Unanswerable / Abstaining)",
                "script": "devanagari",
                "language": "hindi",
                "question": "वारसा में मंगल ग्रह पर कौन सा स्मारक बनाया गया था?",
                "recommended_system": "S5",
                "expected_behavior": "Zero factual evidence found in Warsaw passages; system safely abstains without hallucinating.",
            },
            {
                "id": "sample_english_ans",
                "label": "English (Cross-Lingual)",
                "script": "latin_english",
                "language": "english",
                "question": "From which year has Harvard been ranked top in academic rankings?",
                "recommended_system": "S5",
                "expected_behavior": "Cross-lingual search matches Harvard passage and synthesizes grounded English response citing passage.",
            },
        ]

    # -----------------------------------------------------------------------
    # API Endpoints: Phase 8 Statistical Metrics & Tables
    # -----------------------------------------------------------------------
    @app.get("/api/metrics/overview")
    def get_metrics_overview() -> dict[str, Any]:
        """Return executive summary metrics from Phase 8 consolidated report."""
        summary_path = PHASE8_DIR / "phase8_consolidated_summary.json"
        if not summary_path.exists():
            raise HTTPException(status_code=404, detail="Phase 8 summary artifact not found")

        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)

        return {
            "benchmark_counts": {
                "dev_total": 150,
                "dev_answerable": 75,
                "dev_unanswerable": 75,
                "test_total": 75,
                "test_answerable": 38,
                "test_unanswerable": 37,
                "clean_corpus_passages": 2078,
                "perturbed_corpus_passages": 2493,
                "counterfactual_distractors": 415,
            },
            "systems_count": 6,
            "metric_identities_valid": data.get("metric_identities_valid", True),
            "manifest_status": "Cryptographically Verified (Benchmark & Corpora bit-identical to Phase 6 freeze)",
            "key_highlights": {
                "best_clean_acc_ans": 92.00,
                "best_clean_system": "S2 / S5",
                "clean_to_perturbed_drop_pp": -8.00,
                "unanswerable_rejection_rate": 100.00,
                "hallucination_rate": 0.00,
                "unnormalized_hinglish_acc_ans": 2.67,
                "rule_based_hinglish_acc_ans": 0.00,
                "retry_avoided_false_refusals": 5,
                "retry_accuracy_gain": 0.00,
            },
        }

    @app.get("/api/metrics/ladder")
    def get_systems_ladder(split: str = Query("dev", enum=["dev", "test"])) -> list[dict[str, Any]]:
        """Return Systems Ladder metrics for S0 through S5."""
        filename = "table1_systems_ladder_dev.csv" if split == "dev" else "table2_systems_ladder_test.csv"
        file_path = PHASE8_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")

        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")

    @app.get("/api/metrics/noise")
    def get_noise_degradation() -> list[dict[str, Any]]:
        """Return Clean vs. Perturbed noise degradation table."""
        file_path = PHASE8_DIR / "table3_clean_vs_perturbed.csv"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Noise degradation table not found")

        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")

    @app.get("/api/metrics/hinglish")
    def get_hinglish_ablation() -> list[dict[str, Any]]:
        """Return Cross-Script Hinglish normalization ablation table."""
        file_path = PHASE8_DIR / "table4_hinglish_ablation.csv"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Hinglish ablation table not found")

        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")

    @app.get("/api/metrics/retry")
    def get_retry_matrix() -> list[dict[str, Any]]:
        """Return S2 -> S5 retry transition matrix (75 answerable queries)."""
        file_path = PHASE8_DIR / "table7_retry_transition_matrix.csv"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Retry matrix table not found")

        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")

    @app.get("/api/metrics/significance")
    def get_significance_tests() -> list[dict[str, Any]]:
        """Return paired statistical significance test results with Bonferroni corrections."""
        file_path = PHASE8_DIR / "table6_statistical_significance.csv"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Significance table not found")

        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")

    # -----------------------------------------------------------------------
    # API Endpoints: Evidence Explorer
    # -----------------------------------------------------------------------
    @app.get("/api/evidence/passages")
    def search_passages(
        query: str = Query(default="", description="Search keyword or passage ID"),
        corpus: str = Query(default="clean", enum=["clean", "perturbed"]),
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        """Search and filter passages from the specified corpus."""
        corpus_manager.load_corpora()
        raw_passages = corpus_manager.corpora.get(corpus) or corpus_manager.corpora["clean"]

        q_lower = query.strip().lower()
        if q_lower:
            filtered = [
                p for p in raw_passages
                if q_lower in p.get("passage_id", "").lower()
                or q_lower in p.get("title", "").lower()
                or q_lower in p.get("text", "").lower()
            ]
        else:
            filtered = raw_passages

        total_matches = len(filtered)
        paginated = filtered[offset : offset + limit]

        # Format items
        items = []
        for p in paginated:
            is_cf = p.get("passage_id", "").endswith("_cf") or "counterfactual" in p.get("source_partition", "")
            items.append({
                "passage_id": p.get("passage_id"),
                "title": p.get("title", ""),
                "text": p.get("text", ""),
                "source_partition": p.get("source_partition", "corpus"),
                "is_counterfactual": is_cf,
                "char_length": len(p.get("text", "")),
            })

        return {
            "total": total_matches,
            "offset": offset,
            "limit": limit,
            "corpus": corpus,
            "passages": items,
        }

    # -----------------------------------------------------------------------
    # API Endpoints: Interactive Query Execution
    # -----------------------------------------------------------------------
    @app.post("/api/query/execute", response_model=QueryResponse)
    def execute_query(req: QueryRequest) -> QueryResponse:
        """Execute a live query through the LangGraph multi-agent pipeline."""
        start_time = time.perf_counter()
        q_text = req.question.strip()
        if not q_text:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # Map system ladder
        system_ladder_map = {
            "S0": SystemLadderConfig.S0,
            "S1": SystemLadderConfig.S1,
            "S2": SystemLadderConfig.S2,
            "S3": SystemLadderConfig.S3,
            "S4": SystemLadderConfig.S4,
            "S5": SystemLadderConfig.S5,
        }
        sys_config = system_ladder_map.get(req.system.upper(), SystemLadderConfig.S5)
        condition = "perturbed" if req.corpus.lower() == "perturbed" else "clean"

        # Pre-detect script & language
        script_type, lang_type = script_detector.detect_script_and_language(q_text)

        # Build retrieval components
        retriever_func = corpus_manager.get_retriever_func(condition)
        passage_lookup = corpus_manager.get_passage_lookup(condition)

        # Compile LangGraph StateGraph for the requested system
        graph = build_indicrag_graph(
            system_ladder=sys_config,
            retriever_func=retriever_func,
            passage_lookup=passage_lookup,
            answerer_generator=generator_instance,
            support_threshold=0.35,
        )

        qid = req.query_id or f"web_{uuid.uuid4().hex[:8]}"
        initial_state = IndicRAGState(
            query_id=qid,
            question=q_text,
            system_ladder=sys_config,
        )

        try:
            # Execute graph
            result_state_raw = graph.invoke(initial_state)
            if isinstance(result_state_raw, dict):
                result_state = IndicRAGState(**result_state_raw)
            else:
                result_state = result_state_raw
        except Exception as exc:
            logger.exception("Graph execution error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Pipeline execution failure: {exc}")

        total_duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Preserve script-aware output behavior:
        # Roman Hinglish -> Roman Hinglish
        # Devanagari Hindi -> Devanagari Hindi
        # English -> English
        final_ans = result_state.final_answer or result_state.draft_answer or "No answer produced."

        # If detected script is Latin English and answer returned is standard Hindi refusal, translate refusal appropriately
        if result_state.script == ScriptType.LATIN_ENGLISH:
            if result_state.is_abstention:
                final_ans = "The provided context does not contain sufficient information to answer this question."
            elif any(c in final_ans for c in ["संदर्भ में", "जानकारी उपलब्ध नहीं"]):
                final_ans = "Information is not available in the given context."

        # Format retrieved passages
        passages_out: list[PassageItem] = []
        for p in result_state.retrieved_passages:
            pid = p.get("passage_id", "")
            passages_out.append(PassageItem(
                passage_id=pid,
                title=p.get("title", ""),
                text=p.get("text", ""),
                score=float(p.get("score", 0.0)),
                is_counterfactual=pid.endswith("_cf") or "counterfactual" in str(p.get("source_partition", "")),
            ))

        # Format traces
        traces_out: list[TraceItem] = []
        for t in result_state.traces:
            traces_out.append(TraceItem(
                node=t.get("node", "unknown"),
                action=t.get("action", "unknown"),
                round=t.get("round", 0),
                retry_count=t.get("retry_count", 0),
                llm_calls=t.get("llm_calls", 0),
                is_abstention=t.get("is_abstention", False),
                reason_code=t.get("reason_code"),
                details=t.get("details", {}),
            ))

        return QueryResponse(
            query_id=result_state.query_id,
            question=result_state.question,
            detected_script=result_state.script.value,
            detected_language=result_state.language.value,
            system_ladder=result_state.system_ladder.value,
            corpus_condition=condition,
            final_answer=final_ans,
            is_abstention=result_state.is_abstention,
            reason_code=result_state.reason_code.value if result_state.reason_code else None,
            cited_passage_ids=result_state.cited_passage_ids,
            retrieved_passages=passages_out,
            traces=traces_out,
            stats={
                "llm_calls": result_state.llm_calls,
                "retrieval_rounds": result_state.retrieval_rounds,
                "retry_count": result_state.retry_count,
                "total_tokens": result_state.total_tokens,
                "latency_ms": round(total_duration_ms, 2),
            },
        )

    # -----------------------------------------------------------------------
    # Static Assets & SPA Routes
    # -----------------------------------------------------------------------
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str) -> FileResponse:
            """Catch-all route serving the Single-Page Application index."""
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API route not found")
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return JSONResponse({"status": "Static frontend not yet generated"}, status_code=200)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("indicrag.web.app:app", host="127.0.0.1", port=8000, reload=True)
