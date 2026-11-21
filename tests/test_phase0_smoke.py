"""
Phase 0 smoke test: verify core imports and typed state work.
All checks use stubs — no models, no network calls.
"""
from typing import TypedDict


def test_pydantic_state_schema():
    """Pydantic BaseModel can represent the shared agent state."""

    from pydantic import BaseModel

    class AgentState(BaseModel):
        question: str
        script: str | None = None
        queries: list[str] = []
        retrieved: list[dict] = []
        graded: list[dict] = []
        draft_answer: str | None = None
        citations: list[str] = []
        verdict: str | None = None
        decision: str | None = None
        reason_code: str | None = None
        rounds: int = 0
        calls: int = 0
        tokens: int = 0
        latency: float = 0.0

    state = AgentState(question="भारत की राजधानी क्या है?")
    assert state.question
    assert state.calls == 0
    assert state.rounds == 0


def test_langgraph_stub_graph():
    """LangGraph StateGraph can be built and invoked with a stub node."""
    from langgraph.graph import END, StateGraph

    class StubState(TypedDict):
        value: str
        calls: int

    graph = StateGraph(StubState)

    def stub_agent(state: StubState) -> StubState:
        return {"value": state["value"] + "_answered", "calls": state["calls"] + 1}

    graph.add_node("stub_agent", stub_agent)
    graph.set_entry_point("stub_agent")
    graph.add_edge("stub_agent", END)
    app = graph.compile()

    result = app.invoke({"value": "test_question", "calls": 0})
    assert result["calls"] == 1
    assert "answered" in result["value"]


def test_budget_enforcement():
    """Supervisor must stop when call budget is exceeded."""
    max_calls = 8

    class FakeSupervisor:
        def next_node(self, calls: int) -> str:
            if calls >= max_calls:
                return "END"
            return "continue"

    sv = FakeSupervisor()
    assert sv.next_node(8) == "END"
    assert sv.next_node(7) == "continue"


def test_no_leakage_stub():
    """Article-level split: same article must not appear in both train and test."""
    train_articles = {"India", "Pakistan", "China"}
    test_articles = {"Bangladesh", "Sri Lanka"}
    overlap = train_articles & test_articles
    assert overlap == set(), f"Leakage detected: {overlap}"


def test_reason_codes():
    """All four reason codes are valid strings matching the spec."""
    VALID_CODES = {"NO_EVIDENCE", "UNSUPPORTED_ANSWER", "ANSWERER_REFUSED", "LOW_CONFIDENCE"}
    for code in VALID_CODES:
        assert isinstance(code, str)
        assert len(code) > 0
