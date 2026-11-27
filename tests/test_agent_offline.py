"""Runs the full LangGraph workflow without an API key, using a scripted model.

The scripted model plays the language model's part (choosing tools and writing
text), so these tests check the agent's control flow: routing, tool execution,
the Verifier's revise loop and the guardrails.
"""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.graph import ask


class ScriptedLLM(BaseChatModel):
    script: list = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw: Any) -> ChatResult:
        msg = self.script[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, i=0):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}])


def test_revise_loop_catches_invented_number():
    llm = ScriptedLLM(script=[
        call("compare_groups", {"variables": ["radius"]}),
        AIMessage(content="Malignant tumours have a mean radius of 19.2, benign 12.15."),  # 19.2 is wrong
        AIMessage(content="Malignant samples have a mean radius of 17.46 and benign samples 12.15."),
        AIMessage(content="Cancerous samples tend to be clearly larger than non-cancerous ones."),
    ])
    out = ask("Compare the average radius of benign and malignant cases.", llm=llm, save_log=False)
    events = [(e["agent"], e["event"]) for e in out["trace"]]
    assert ("Verifier", "revise") in events and ("Verifier", "passed") in events
    assert "17.46" in out["answer"] and out["verified"]
    assert out["tables"][0]["title"] == "Comparison by diagnosis"


def test_simulated_age_disclosure_enforced():
    llm = ScriptedLLM(script=[
        call("describe_variable", {"variable": "age_simulated",
                                   "filters": [{"column": "diagnosis", "op": "==", "value": "Malignant"}]}),
        AIMessage(content="The average age of malignant patients is 57.82."),            # no disclosure
        AIMessage(content="The average simulated age of malignant cases is 57.82 (age is simulated)."),
        AIMessage(content="Age here was generated at random, so it says nothing medical."),
    ])
    out = ask("What is the average age of malignant patients?", llm=llm, save_log=False)
    assert any("SIMULATED" in n for n in out["notes"])
    assert "simulated" in out["answer"].lower()


def test_survival_not_in_dataset():
    llm = ScriptedLLM(script=[
        call("describe_variable", {"variable": "survival"}),
        AIMessage(content="The dataset does not contain survival information, so I cannot compute a five-year survival rate."),
    ])
    out = ask("What is the five-year survival rate in this data?", llm=llm, save_log=False)
    assert "does not contain" in out["answer"] and out["explanation"] == ""


def test_personal_medical_request_refused_without_llm():
    llm = ScriptedLLM(script=[])  # must not be called
    out = ask("I found a lump, do I have breast cancer?", llm=llm, save_log=False)
    assert "can't give personal medical advice" in out["answer"] and llm.calls == 0


def test_explainer_new_numbers_dropped():
    llm = ScriptedLLM(script=[
        call("count_cases", {"group_by": ["diagnosis"]}),
        AIMessage(content="212 cases (37.3%) are malignant and 357 (62.7%) are benign."),
        AIMessage(content="Roughly 4 in 10 are malignant, similar to the 45% seen in clinics."),  # 45 invented
    ])
    out = ask("How many cases are malignant and benign?", llm=llm, save_log=False)
    assert out["explanation"] == ""
    assert ("Medical Explainer", "dropped_new_numbers") in [(e["agent"], e["event"]) for e in out["trace"]]


def test_step_limit_forces_final_answer():
    loop = [call("describe_variable", {"variable": "radius"}, i) for i in range(6)]
    llm = ScriptedLLM(script=loop + [
        AIMessage(content="The mean radius is 14.1273."),            # forced answer at the limit
        AIMessage(content="Radius measures the size of the cell nucleus."),
    ])
    out = ask("What is the mean radius?", llm=llm, save_log=False)
    assert out["answer"] == "The mean radius is 14.1273." and out["verified"]
    assert ("Data Analyst", "draft_answer_at_step_limit") in [(e["agent"], e["event"]) for e in out["trace"]]


def test_empty_model_answer_gets_fallback():
    llm = ScriptedLLM(script=[call("count_cases", {}), AIMessage(content=""), AIMessage(content="")])
    out = ask("Count the cases", llm=llm, save_log=False)
    assert out["answer"].startswith("Sorry, I could not produce an answer")
