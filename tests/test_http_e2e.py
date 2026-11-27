"""End-to-end: ask() -> real ChatOpenAI client -> HTTP -> OpenAI-compatible mock server.

Runs every Section 6 question (T1-T10, D1-D10) and the guardrail questions through
the full agent, exactly as the web interface will call it.
"""
import os

import pytest

import agent.graph as G
from mock_openai_server import Handler, start

# question, strings that must appear in the answer
CASES = [
    ("T1", "How many records and columns are in the dataset?", ["rows = 569", "columns = 14"]),
    ("T2", "What are the column names?", ["columns = 14"]),
    ("T3", "Are there missing values in any column?", ["total_missing = 0"]),
    ("T4", "Are there duplicate rows?", ["duplicate_rows = 0"]),
    ("T5", "What is the data type of each column?", ["columns = 14"]),
    ("T6", "What are the minimum, maximum and mean of radius?", ["min = 6.981", "max = 28.11", "mean = 14.1273"]),
    ("T7", "Show the distribution of area.", ["median = 551.1", "max = 2501.0", "min = 143.5"]),
    ("T8", "Compute the correlation matrix of the numeric measurements.", ["strongest_pairs"]),
    ("T9", "Identify outliers in texture using the IQR rule.", ["n_outliers = 7"]),
    ("T10", "Compare the standard deviation of radius and area and explain the difference.", ["std = 3.524", "std = 351.9141"]),
    ("D1", "What percentage of cases are malignant and benign?", ["percent = 62.7", "percent = 37.3"]),
    ("D2", "How many cases are malignant and how many benign?", ["count = 357", "count = 212"]),
    ("D3", "What is the average radius overall?", ["mean = 14.1273"]),
    ("D4", "What are the largest and smallest area values?", ["max = 2501.0", "min = 143.5"]),
    ("D5", "Compare the average radius of benign and malignant cases.", ["mean = 12.1465", "mean = 17.4628"]),
    ("D6", "Compare texture between the two diagnoses.", ["mean = 17.9148", "mean = 21.6049"]),
    ("D7", "Show the distribution of area for each diagnosis.", ["Benign.median = 458.4", "Malignant.median = 932.0"]),
    ("D8", "Which variables are most strongly associated with the diagnosis?", ["r = 0.78"]),
    ("D9", "For a new case with radius 17 and concave points 0.08, is the tumour more likely malignant or benign, and why?", ["probability_malignant = 0.931"]),
    ("D10", "Summarise in plain language how malignant tumours differ from benign ones in this data.", ["Benign.median = 0.0234", "Malignant.median = 0.0862"]),
]


@pytest.fixture(scope="module")
def server():
    srv, url = start()
    old = {k: os.environ.get(k) for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL", "NO_PROXY", "no_proxy")}
    os.environ.update(OPENAI_API_KEY="test-key", OPENAI_BASE_URL=url, LLM_MODEL="mock-model",
                      NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    G._GRAPH = None
    yield url
    srv.shutdown()
    G._GRAPH = None
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.parametrize("qid,question,expected", CASES, ids=[c[0] for c in CASES])
def test_section6_question(server, qid, question, expected, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    monkeypatch.setattr("agent.tools.CHART_DIR", tmp_path / "charts")
    out = G.ask(question)                       # the exact call the interface makes
    for s in expected:
        assert s in out["answer"], f"{qid}: '{s}' missing from: {out['answer']}"
    assert out["verified"] is True
    assert out["tools_used"], "no tool was used"
    assert out["notes"][-1].startswith("Educational project")
    assert set(out) >= {"answer", "explanation", "notes", "tables", "charts", "verified", "tools_used", "trace"}
    assert any(f.endswith(".jsonl") for f in os.listdir(tmp_path))  # log written
    if qid in {"T7", "D7"}:
        assert out["charts"] and os.path.exists(out["charts"][0])
    if qid == "T10":
        assert [t["name"] for t in out["tools_used"]] == ["describe_variable", "describe_variable"]
    if qid == "D9":
        assert any("NOT a diagnosis" in n for n in out["notes"])


def test_age_question_discloses_simulation(server, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    out = G.ask("What is the average age of malignant patients?")
    assert "mean = 57.8208" in out["answer"] and "simulated" in out["answer"].lower()
    assert any("SIMULATED" in n for n in out["notes"])


def test_filter_question(server, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    out = G.ask("How many cases are over 60, by diagnosis?")
    assert "count = 157" in out["answer"] and "count = 87" in out["answer"]


def test_survival_not_in_dataset(server, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    out = G.ask("What is the five-year survival rate in this data?")
    assert "does not contain" in out["answer"] and out["explanation"] == ""


def test_medical_advice_refused_without_model_call(server, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    n = len(Handler.requests_seen)
    out = G.ask("I found a lump, do I have breast cancer?")
    assert "can't give personal medical advice" in out["answer"]
    assert len(Handler.requests_seen) == n


def test_conversation_history_is_sent(server, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    first = G.ask("How many cases are malignant and how many benign?")
    G.ask("And what percentage is that?", history=[{"question": "How many cases are malignant and how many benign?",
                                                     "answer": first["answer"]}])
    roles = [m["role"] for m in Handler.requests_seen[-2]["messages"]]
    assert roles[:4] == ["system", "user", "assistant", "user"]


def test_every_request_was_valid(server):
    # the mock answers 400 to malformed requests; ask() would have raised, but check anyway
    assert Handler.requests_seen and all(r["model"] == "mock-model" for r in Handler.requests_seen)
    tools = Handler.requests_seen[0]["tools"]
    assert sorted(t["function"]["name"] for t in tools) == sorted([
        "dataset_overview", "describe_variable", "count_cases", "compare_groups",
        "correlation", "find_outliers", "make_chart", "predict_case"])


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="No API key found"):
        G.get_llm()


@pytest.mark.parametrize("model,has_temp", [("gpt-5.4-mini", False), ("gpt-4o-mini", True),
                                            ("llama-3.3-70b-versatile", True)])
def test_temperature_only_sent_when_supported(server, model, has_temp, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    monkeypatch.setenv("LLM_MODEL", model)
    G._GRAPH = None
    out = G.ask("How many cases are malignant and how many benign?")
    assert "count = 212" in out["answer"]
    req = Handler.requests_seen[-1]
    assert req["model"] == model and (("temperature" in req) == has_temp)
    G._GRAPH = None


def test_default_model_is_current():
    assert G.DEFAULT_MODEL == "gpt-5.4-mini"


def test_step_limit_over_http(server, tmp_path, monkeypatch):
    monkeypatch.setattr(G, "LOG_DIR", tmp_path)
    out = G.ask("Please keep calling tools about radius.")
    events = [e["event"] for e in out["trace"]]
    assert events.count("select_tools") == G.MAX_TOOL_ROUNDS
    assert "draft_answer_at_step_limit" in events and "mean = 14.1273" in out["answer"]
    final_req = [r for r in Handler.requests_seen if r.get("tool_choice") == "none"][-1]
    assert final_req["tools"]  # tools still declared, so the tool-call history is valid
