# Agentic AI for Breast Cancer Data Analysis: agent and web interface

A LangGraph multi-agent system. It answers free-text questions about
`data/final_dataset.csv` (569 rows x 14 columns) by choosing and running data tools.
It does not answer from the language model's memory. `app.py` is a ready-to-use
Streamlit interface for it.

## 1. Install (Python 3.11 or newer)

```bash
pip install -r requirements.txt
```

## 2. Add your API key

Copy `.env.example` to `.env` and put your key in it:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-5.4-mini
```

You can also paste the key in the app's sidebar. Other OpenAI-compatible providers
(Groq, Gemini, Ollama) work too; see `.env.example`. The model must support tool
calling. Do not share or submit your `.env` file.

## 3. Run

```bash
streamlit run app.py                  # web interface (opens in the browser)
python -m agent "Compare the average radius of benign and malignant cases"   # one question
python -m agent                       # chat in the terminal
python -m pytest -q                   # 96 tests, no API key or internet needed
```

## Using the agent from another interface

```python
from agent import ask

out = ask("Show the distribution of area for each diagnosis", history=[])
out["answer"]        # Retriever's verified answer (markdown text)
out["explanation"]   # Medical Explainer's plain-language text ("" if none)
out["notes"]         # simulated-data note, prediction warning, disclaimer
out["tables"]        # [{"title": str, "rows": [dict, ...]}] -> pd.DataFrame(t["rows"])
out["charts"]        # list of PNG file paths
out["tools_used"]    # tool names and arguments
out["trace"]         # step-by-step log of the agentic loop
out["verified"]      # True if every number matched the tool output
```

`history` is a list of earlier turns: `[{"question": ..., "answer": ...}]`.
Every run is added to `logs/YYYY-MM-DD.jsonl`; Section 4.5 and Section 6 use these logs.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web interface |
| `agent/tools.py` | 8 pandas/scikit-learn data tools and the code-level guardrails |
| `agent/lc_tools.py` | Tool schemas the language model sees |
| `agent/graph.py` | LangGraph workflow: Coordinator, Retriever, Tools, Verifier, Medical Explainer |
| `agent/verify.py` | Number check, simulated-data check, medical-advice filter |
| `agent/prompts.py` | System prompts, refusal text, disclaimer |
| `data/final_dataset.csv` | The prepared dataset (Section 3) |
| `tests/` | Tool outputs vs. Section 6 ground truth; full loop with a scripted model; all 20 Section 6 questions over HTTP against a local OpenAI-compatible test server; the Streamlit app |
| `docs/architecture.png` | Figure 7 for Section 4.2 |

## Note on testing

The automated tests check everything except the language model's own reasoning:
the numbers, the tools, the agent loop, the guardrails, the API requests and the app.
They use a local test server in place of the real API. Before submitting, run the
Section 6 questions with a real API key and record the agent's answers in Section 6.

## Deploy online (Render)

The repository includes `render.yaml`. On render.com choose **New → Blueprint**, connect the
GitHub repository and enter your Groq key when Render asks for `OPENAI_API_KEY`.
`PUBLIC_MODE=1` hides the Settings box, so visitors cannot change the key, model or provider.
Never upload `.env`: `.gitignore` already excludes it.
