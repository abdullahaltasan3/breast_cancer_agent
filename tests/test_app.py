"""Runs the Streamlit app headlessly against the mock OpenAI server."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from mock_openai_server import start  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


@pytest.fixture(autouse=True)
def isolate_outputs(monkeypatch, tmp_path):
    import agent.graph as G
    import agent.tools as T
    monkeypatch.setattr(G, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(T, "CHART_DIR", tmp_path / "charts")


@pytest.fixture()
def mock_env(monkeypatch):
    srv, url = start()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.setenv("LLM_MODEL", "mock-model")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    import agent.graph as G
    G._GRAPH = None
    yield
    srv.shutdown()
    G._GRAPH = None


def _text(at):
    return " ".join(m.value for m in at.markdown)


def test_app_loads_with_disclaimer_and_dataset_connection(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "Not for medical use" in _text(at)
    assert "data assistant" in _text(at) and "orb" in _text(at)
    assert "Connected to" in _text(at) and "569 records" in _text(at)
    assert all(at.button(key=f"ex{i}") for i in range(6))  # suggestion buttons


def test_app_answers_with_table_chart_and_steps(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("Show the distribution of area for each diagnosis.").run()
    assert not at.exception
    assert "Malignant.median = 932.0" in _text(at)
    assert "🔎 Retriever" in _text(at) and "✅ Verifier" in _text(at)   # the 4-agent strip
    assert len(at.get("image")) + len(at.get("imgs")) >= 1  # name differs by Streamlit version
    assert not any(b.label.startswith("Show the distribution") for b in at.button)  # examples hidden
    # second turn sends history and still works
    at.chat_input[0].set_value("Compare the average radius of benign and malignant cases.").run()
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert len(at.chat_message) == 4


def test_app_example_button(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="ex0").click().run()
    assert not at.exception
    assert "count = 357" in _text(at)


def test_app_new_chat_keeps_old_chat(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("How many cases are malignant and how many benign?").run()
    at.button(key="new_chat").click().run()
    assert not at.exception
    assert len(at.session_state["chats"]) == 2 and len(at.chat_message) == 0


def test_app_shows_error_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    import agent.graph as G
    G._GRAPH = None
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("How many cases are malignant?").run()
    assert not at.exception
    assert "No API key found" in _text(at)


def test_app_gender_and_refusal_suggestions(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="ex1").click().run()                       # females / males
    assert not at.exception and "count = 564" in _text(at) and "count = 5" in _text(at)
    at.button(key="new_chat").click().run()
    at.button(key="ex2").click().run()                       # I have a small tumour
    assert not at.exception and "not a diagnosis" in _text(at)
    assert "Safety check" in _text(at) or "safety check" in _text(at)


def test_public_mode_hides_settings(mock_env, monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.text_input) == 0          # no key / model / base URL boxes for visitors
    at.button(key="ex0").click().run()      # the agent still answers
    assert not at.exception and "count = 357" in _text(at)
