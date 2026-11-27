"""Multi-agent workflow built with LangGraph.

    question
       |
  [Coordinator: intake] --personal medical request--> [Coordinator: final] (refusal)
       |
  [Retriever] <---------------------------+
       | tool calls?                         | revise (max 2)
       v yes                 no              |
   [Tools] --> back to Retriever   [Verifier] -+
                                    | passed
                             [Medical Explainer]
                                    |
                           [Coordinator: final] --> answer + tables + charts + trace
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .lc_tools import LC_TOOLS, LC_TOOLS_BY_NAME
from .prompts import (RETRIEVER_PROMPT, DISCLAIMER, EXPLAINER_PROMPT, REFUSAL_TEXT,
                      REFUSAL_TEXT_AR, REVISION_PROMPT, SAFETY_FOOTER, SAFETY_FOOTER_AR,
                      SUPPORT_DATA_PROMPT, SUPPORT_FALLBACK, SUPPORT_FALLBACK_AR, SUPPORT_PROMPT)
from .tools import SIMULATED_NOTE
from .verify import (is_arabic, is_personal_medical_request, mentions_simulated, numbers_in_text, unsupported_numbers,
                     used_simulated)

DEFAULT_MODEL = "gpt-5.4-mini"  # gpt-4o-mini still works but OpenAI retires it on 14 Jan 2027
MAX_TOOL_ROUNDS = 6
MAX_REVISIONS = 2
LOG_DIR = Path(os.getenv("LOG_DIR", Path(__file__).resolve().parent.parent / "logs"))


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    messages: Annotated[list, add_messages]   # Retriever's working conversation
    tool_calls: list[dict]                    # every call: name, args, result
    tool_rounds: int
    draft: str
    revisions: int
    verified: bool
    problems: list[str]
    explanation: str
    refused: bool
    mode: str                                 # "data" or "support" (personal health question)
    final: dict
    trace: list[dict]


def _log(state: AgentState, agent: str, event: str, **detail) -> list[dict]:
    entry = {"t": round(time.time(), 3), "agent": agent, "event": event, **detail}
    return (state.get("trace") or []) + [entry]


def _text(msg) -> str:
    """Message content as plain text (some providers return a list of parts)."""
    c = getattr(msg, "content", msg)
    if isinstance(c, list):
        c = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return (c or "").strip()


def get_llm():
    """Any OpenAI-compatible chat model. Configure with environment variables:
    OPENAI_API_KEY, LLM_MODEL (default gpt-5.4-mini) and optionally OPENAI_BASE_URL
    (e.g. Groq, OpenRouter, Gemini's OpenAI endpoint or a local Ollama server)."""
    from langchain_openai import ChatOpenAI
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # project .env
        load_dotenv()  # and a .env in the current folder, if any
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "No API key found. Put OPENAI_API_KEY=... in a .env file next to README.md "
            "(or set it as an environment variable) and restart.")
    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    kwargs = {}
    if os.getenv("LLM_TEMPERATURE"):
        kwargs["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
    elif not re.match(r"^(gpt-5|gpt-6|o\d)", model):
        kwargs["temperature"] = 0  # reasoning models (gpt-5.x, o-series) only accept the default
    return ChatOpenAI(model=model, base_url=os.getenv("OPENAI_BASE_URL") or None,
                      timeout=float(os.getenv("LLM_TIMEOUT", "120")), max_retries=2, **kwargs)


def build_graph(llm=None):
    llm = llm or get_llm()
    retriever_llm = llm.bind_tools(LC_TOOLS)
    try:  # same tools, but the model may not call them (used when the step limit is reached)
        final_llm = llm.bind_tools(LC_TOOLS, tool_choice="none")
    except Exception:  # noqa: BLE001 - providers without tool_choice support
        final_llm = llm

    # ---------------- Coordinator (intake) ----------------
    def intake(state: AgentState):
        q = state["question"]
        if is_personal_medical_request(q):
            return {"mode": "support", "refused": False, "tool_calls": [], "messages": [],
                    "trace": _log(state, "Coordinator", "support_mode_personal_health_question")}
        msgs = [SystemMessage(RETRIEVER_PROMPT)]
        for turn in (state.get("history") or [])[-3:]:
            msgs += [HumanMessage(turn["question"]), AIMessage(turn["answer"])]
        msgs.append(HumanMessage(q))
        return {"messages": msgs, "tool_calls": [], "tool_rounds": 0, "revisions": 0, "mode": "data",
                "refused": False, "trace": _log(state, "Coordinator", "route_to_retriever", question=q)}

    # ---------------- Retriever ----------------
    def retriever(state: AgentState):
        if state.get("tool_rounds", 0) >= MAX_TOOL_ROUNDS:
            # step limit reached: ask for a final answer with the results so far, no more tools
            ai = final_llm.invoke(state["messages"] + [HumanMessage(
                "Tool limit reached. Answer now using only the tool results above.")])
            ai = AIMessage(content=_text(ai))
            return {"messages": [ai], "draft": ai.content,
                    "trace": _log(state, "Retriever", "draft_answer_at_step_limit", draft=ai.content)}
        ai = retriever_llm.invoke(state["messages"])
        if ai.tool_calls:
            plan = [{"tool": c["name"], "args": c["args"]} for c in ai.tool_calls]
            return {"messages": [ai], "trace": _log(state, "Retriever", "select_tools", plan=plan)}
        return {"messages": [ai], "draft": _text(ai),
                "trace": _log(state, "Retriever", "draft_answer", draft=_text(ai))}

    # ---------------- Retriever (support mode) ----------------
    def support_retriever(state: AgentState):
        """For a personal health question: fetch general facts from the dataset (fixed, safe
        tool calls) and write a short, clearly-labelled summary of what the research data shows."""
        plan = [("count_cases", {"group_by": ["diagnosis"]}),
                ("compare_groups", {"variables": ["radius", "concave_points"]})]
        trace = _log(state, "Retriever", "select_tools", plan=[{"tool": n, "args": a} for n, a in plan])
        calls = []
        for name, args in plan:
            result = LC_TOOLS_BY_NAME[name].invoke(args)
            calls.append({"name": name, "args": args, "result": result})
            trace = _log({"trace": trace}, "Tools", "result", tool=name, ok="error" not in result,
                         preview=json.dumps(result, default=str)[:300])
        compact = json.dumps([{"tool": c["name"], "result": {k: v for k, v in c["result"].items()
                                                              if k not in {"notes", "source"}}}
                              for c in calls], default=str)
        try:
            draft = _text(llm.invoke([HumanMessage(SUPPORT_DATA_PROMPT.format(
                question=state["question"], results=compact))]))
        except Exception:  # noqa: BLE001
            draft = ""
        return {"tool_calls": calls, "draft": draft,
                "trace": _log({"trace": trace}, "Retriever", "draft_data_summary", draft=draft)}

    def run_tools(state: AgentState):
        ai: AIMessage = state["messages"][-1]
        new_msgs, calls, trace = [], list(state.get("tool_calls") or []), state.get("trace")
        for c in ai.tool_calls:
            result = LC_TOOLS_BY_NAME[c["name"]].invoke(c["args"]) if c["name"] in LC_TOOLS_BY_NAME \
                else {"error": f"Unknown tool {c['name']}"}
            calls.append({"name": c["name"], "args": c["args"], "result": result})
            new_msgs.append(ToolMessage(json.dumps(result, default=str), tool_call_id=c["id"], name=c["name"]))
            trace = _log({"trace": trace}, "Tools", "result", tool=c["name"],
                         ok="error" not in result, preview=json.dumps(result, default=str)[:300])
        return {"messages": new_msgs, "tool_calls": calls,
                "tool_rounds": state.get("tool_rounds", 0) + 1, "trace": trace}

    def after_retriever(state: AgentState):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "verifier"

    # ---------------- Verifier (checks the result) ----------------
    def verifier(state: AgentState):
        draft = state.get("draft") or ""
        calls = state.get("tool_calls") or []
        results = [c["result"] for c in calls]
        problems = []
        bad = unsupported_numbers(draft, results, state["question"])
        if not calls and bad:
            problems.append("You wrote numbers without calling any tool. Call the right tool first.")
        elif bad:
            problems.append("These numbers do not appear in any tool result: " + ", ".join(bad) +
                            ". Remove them or replace them with numbers from the tools.")
        if used_simulated(calls) and not mentions_simulated(draft):
            problems.append("You used age or gender: state that they are simulated.")
        if state.get("mode") == "support" and problems:
            return {"verified": True, "problems": [], "draft": "",
                    "trace": _log(state, "Verifier", "dropped_data_summary", problems=problems)}
        if not problems:
            return {"verified": True, "problems": [],
                    "trace": _log(state, "Verifier", "passed", numbers_checked=len(numbers_in_text(draft)))}
        if state.get("revisions", 0) >= MAX_REVISIONS:
            return {"verified": False, "problems": problems,
                    "trace": _log(state, "Verifier", "failed_after_max_revisions", problems=problems)}
        return {"revisions": state.get("revisions", 0) + 1, "problems": problems,
                "messages": [HumanMessage(REVISION_PROMPT.format(problems="\n".join(problems)))],
                "trace": _log(state, "Verifier", "revise", problems=problems)}

    def after_verifier(state: AgentState):
        # a revision request was appended -> back to the retriever; otherwise go on
        if state.get("mode") == "support" or not state.get("messages"):
            return "explainer"
        return "retriever" if state["messages"][-1].type == "human" else "explainer"

    # ---------------- Medical Explainer ----------------
    def explainer(state: AgentState):
        if state.get("mode") == "support":
            arabic = is_arabic(state["question"])
            try:
                text = _text(llm.invoke([HumanMessage(SUPPORT_PROMPT.format(question=state["question"]))]))
            except Exception:  # noqa: BLE001
                text = ""
            unsafe = (not text or unsupported_numbers(text, [], state["question"]) or re.search(
                r"\b(you (?:have|don't have|do not have) cancer|it(?: is|'s) (?:benign|malignant)|"
                r"\d+\s*mg|you will (?:die|be fine|survive))\b", text, re.I))
            if unsafe:
                return {"explanation": SUPPORT_FALLBACK_AR if arabic else SUPPORT_FALLBACK,
                        "trace": _log(state, "Medical Explainer", "guidance_fallback")}
            return {"explanation": text, "trace": _log(state, "Medical Explainer", "gave_guidance")}
        calls = state.get("tool_calls") or []
        if not any("error" not in c["result"] for c in calls):
            return {"explanation": "", "trace": _log(state, "Medical Explainer", "skipped_no_data")}
        msg = llm.invoke([HumanMessage(EXPLAINER_PROMPT.format(
            question=state["question"], answer=state.get("draft", "")))])
        text = _text(msg)
        bad = unsupported_numbers(text, [c["result"] for c in calls] + [state.get("draft", "")],
                                  state["question"])
        if bad:  # guardrail: the explainer may not introduce new figures
            return {"explanation": "", "trace": _log(state, "Medical Explainer", "dropped_new_numbers", numbers=bad)}
        return {"explanation": text, "trace": _log(state, "Medical Explainer", "explained")}

    # ---------------- Coordinator (final answer) ----------------
    def finalize(state: AgentState):
        if state.get("mode") == "support":
            arabic = is_arabic(state["question"])
            calls = state.get("tool_calls") or []
            parts = [state.get("explanation") or (SUPPORT_FALLBACK_AR if arabic else SUPPORT_FALLBACK)]
            if state.get("draft"):
                header = "ماذا تُظهر بيانات البحث" if arabic else "What the research data shows"
                parts.append(f"**{header}:** {state['draft']}")
            parts.append(f"*{SAFETY_FOOTER_AR if arabic else SAFETY_FOOTER}*")
            final = {"answer": "\n\n".join(parts), "explanation": "", "notes": [DISCLAIMER],
                     "tables": [],  # no statistics tables in a reply to someone's personal situation
                     "charts": [], "verified": True,
                     "tools_used": [{"name": c["name"], "args": c["args"]} for c in calls]}
            return {"final": final, "trace": _log(state, "Coordinator", "final_answer_support")}
        calls = state.get("tool_calls") or []
        notes = []
        if used_simulated(calls) or any(SIMULATED_NOTE in str(c["result"]) for c in calls):
            notes.append(SIMULATED_NOTE)
        for c in calls:
            if c["name"] == "predict_case" and "error" not in c["result"]:
                notes.extend(c["result"].get("notes", []))
        if state.get("verified") is False:
            notes.append("Warning: some figures in this answer could not be matched to the data tools.")
        notes.append(DISCLAIMER)
        answer = (state.get("draft") or "").strip() or (
            "Sorry, I could not produce an answer to that question. Please rephrase it, "
            "for example: 'Compare the average area of benign and malignant cases.'")
        final = {
            "answer": answer,
            "explanation": state.get("explanation", ""),
            "notes": list(dict.fromkeys(notes)),
            "tables": [t for c in calls if (t := to_table(c)) and wants_table(c, state["question"])],
            "charts": [c["result"]["chart_path"] for c in calls if "chart_path" in c["result"]],
            "verified": bool(state.get("verified")),
            "tools_used": [{"name": c["name"], "args": c["args"]} for c in calls],
        }
        return {"final": final, "trace": _log(state, "Coordinator", "final_answer")}

    g = StateGraph(AgentState)
    g.add_node("intake", intake)
    g.add_node("retriever", retriever)
    g.add_node("tools", run_tools)
    g.add_node("verifier", verifier)
    g.add_node("explainer", explainer)
    g.add_node("finalize", finalize)
    g.add_edge(START, "intake")
    g.add_node("support_retriever", support_retriever)
    g.add_conditional_edges("intake", lambda s: "support_retriever" if s.get("mode") == "support" else "retriever")
    g.add_edge("support_retriever", "verifier")
    g.add_conditional_edges("retriever", after_retriever, {"tools": "tools", "verifier": "verifier"})
    g.add_edge("tools", "retriever")
    g.add_conditional_edges("verifier", after_verifier, {"retriever": "retriever", "explainer": "explainer"})
    g.add_edge("explainer", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


_OVERVIEW_WORDS = re.compile(r"column|field|variable|data ?type|type of each|missing|duplicate|"
                             r"structure|schema|عمود|أعمدة|اعمدة|نوع البيانات|مفقود|مكرر", re.I)


def wants_table(call: dict, question: str) -> bool:
    """The columns overview is only useful when the user asked about the dataset's structure."""
    if call["name"] == "dataset_overview":
        return bool(_OVERVIEW_WORDS.search(question or ""))
    return True


def to_table(call: dict) -> Optional[dict]:
    """Turn a tool result into a simple table (title + list of row dicts) for the UI."""
    r, n = call["result"], call["name"]
    if "error" in r:
        return None
    if n == "describe_variable":
        if "by_group" in r:
            rows = [{"group": g, **s} for g, s in r["by_group"].items()]
        else:
            rows = [{"group": "all", **r.get("overall", {})}]
        return {"title": f"Statistics of {r['variable']}", "rows": rows}
    if n == "count_cases":
        if not r.get("counts"):
            return None  # a single total is already in the answer text; a one-cell table adds nothing
        return {"title": "Case counts", "rows": r["counts"]}
    if n == "compare_groups":
        rows = []
        for item in r["comparison"]:
            for g, s in item.items():
                if isinstance(s, dict) and "mean" in s:
                    rows.append({"variable": item["variable"], "group": g, **s})
        return {"title": f"Comparison by {r['group_by']}", "rows": rows}
    if n == "correlation":
        if "ranking" in r:
            return {"title": "Correlation with malignant diagnosis", "rows": r["ranking"]}
        return {"title": "Correlation matrix",
                "rows": [{"variable": k, **v} for k, v in r["matrix"].items()]}
    if n == "find_outliers":
        if "by_group" in r:
            rows = [{"group": g, **{k: v for k, v in s.items() if k != "outliers"}} for g, s in r["by_group"].items()]
        else:
            rows = r["outliers"]
        return {"title": f"IQR outliers of {r['variable']}", "rows": rows}
    if n == "dataset_overview":
        return {"title": "Columns", "rows": [{"column": c, "type": r["dtypes"][c],
                                              "missing": r["missing_values_per_column"][c]}
                                             for c in r["column_names"]]}
    if n == "predict_case":
        return {"title": "Prediction input", "rows": [{"measurement": k, "value": v}
                                                      for k, v in r["input_used"].items()]}
    return None


_GRAPH = None


def ask(question: str, history: list[dict] | None = None, llm=None, save_log: bool = True) -> dict:
    """Main entry point for the web interface.

    Returns {"answer", "explanation", "notes", "tables", "charts", "verified",
             "tools_used", "trace"}. `history` is a list of {"question", "answer"}.
    """
    global _GRAPH
    graph = build_graph(llm) if llm is not None else (_GRAPH or build_graph())
    if llm is None:
        _GRAPH = graph
    state = graph.invoke({"question": question, "history": history or [], "trace": []},
                         {"recursion_limit": 40})
    out = dict(state["final"])
    out["trace"] = state.get("trace", [])
    if save_log:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"question": question, **{k: v for k, v in out.items()}},
                               default=str) + "\n")
    return out


def format_answer(out: dict) -> str:
    """Plain-text rendering (CLI / simple UIs)."""
    parts = [out["answer"]]
    if out.get("explanation"):
        parts.append("In plain language: " + out["explanation"])
    parts += [f"Note: {n}" for n in out.get("notes", [])]
    parts += [f"Chart saved: {c}" for c in out.get("charts", [])]
    return "\n\n".join(p for p in parts if p)
