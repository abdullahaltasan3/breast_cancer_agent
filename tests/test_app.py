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


def test_app_loads_with_banner(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Not for medical use" in w.value for w in at.warning)
    assert len(at.button) >= 6  # example questions + clear


def test_app_answers_with_table_chart_and_steps(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("Show the distribution of area for each diagnosis.").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Malignant.median = 932.0" in text
    assert len(at.get("image")) + len(at.get("imgs")) >= 1  # name differs by Streamlit version
    assert not any(b.label.startswith("Show the distribution") for b in at.button)  # examples hidden
    # second turn sends history and still works
    at.chat_input[0].set_value("Compare the average radius of benign and malignant cases.").run()
    assert not at.exception
    assert len(at.dataframe) >= 1
    assert len(at.chat_message) == 4


def test_app_example_button(mock_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("count = 357" in m.value for m in at.markdown)


def test_app_shows_error_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    import agent.graph as G
    G._GRAPH = None
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.chat_input[0].set_value("How many cases are malignant?").run()
    assert not at.exception
    assert any("No API key found" in m.value for m in at.markdown)
