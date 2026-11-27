"""A tiny OpenAI-compatible HTTP server for testing the agent without an API key.

It is NOT a language model: it picks tools with keyword rules and writes answers
from the tool results. Its purpose is to test everything around the model
through the real ChatOpenAI client: the tool schemas sent over HTTP, tool-call
ids, tool messages, revisions and parsing of responses. Like the real API, it
rejects malformed requests with HTTP 400.
"""
from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# question keyword -> (tool, args); first match wins
ROUTES = [
    ("people have a malignant tumour", ("count_cases", {"group_by": ["diagnosis"]})),
    ("females", ("count_cases", {"group_by": ["gender_simulated"]})),
    ("over 60", ("count_cases", {"group_by": ["diagnosis"], "filters": [
        {"column": "age_simulated", "op": ">", "value": 60}]})),
    ("survival", ("describe_variable", {"variable": "five-year survival"})),
    ("records and columns", ("dataset_overview", {})),
    ("column names", ("dataset_overview", {})),
    ("missing", ("dataset_overview", {})),
    ("duplicate", ("dataset_overview", {})),
    ("data type", ("dataset_overview", {})),
    ("standard deviation", [("describe_variable", {"variable": "radius"}),   # two calls in one turn
                            ("describe_variable", {"variable": "area"})]),
    ("minimum, maximum and mean of radius", ("describe_variable", {"variable": "radius"})),
    ("distribution of area for each", ("make_chart", {"chart_type": "histogram", "variable": "area", "by_diagnosis": True})),
    ("distribution of area", ("make_chart", {"chart_type": "histogram", "variable": "area", "by_diagnosis": False})),
    ("correlation matrix", ("correlation", {})),
    ("outliers", ("find_outliers", {"variable": "texture"})),
    ("percentage", ("count_cases", {"group_by": ["diagnosis"]})),
    ("how many cases", ("count_cases", {"group_by": ["diagnosis"]})),
    ("average radius overall", ("describe_variable", {"variable": "radius"})),
    ("largest and smallest area", ("describe_variable", {"variable": "area"})),
    ("average radius of benign", ("compare_groups", {"variables": ["radius"]})),
    ("texture between", ("compare_groups", {"variables": ["texture"]})),
    ("most strongly associated", ("correlation", {"with_diagnosis": True})),
    ("new case", ("predict_case", {"radius": 17, "concave_points": 0.08})),
    ("summarise", ("compare_groups", {"variables": ["radius", "area", "concave_points"]})),
    ("average age", ("describe_variable", {"variable": "age", "filters": [
        {"column": "diagnosis", "op": "==", "value": "Malignant"}]})),
]


def _leaves(obj, prefix=""):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"outliers", "matrix", "missing_values_per_column", "dtypes", "input_used"}:
                continue
            yield from _leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:6]):
            yield from _leaves(v, f"{prefix}[{i}]")


def _answer_from_tools(tool_msgs: list[dict]) -> str:
    parts, simulated = [], False
    for m in tool_msgs:
        r = json.loads(m["content"])
        if r.get("error_type") == "not_in_dataset":
            return "The dataset does not contain that information, so I cannot answer it from the data."
        simulated |= "SIMULATED" in m["content"]
        facts = [f"{k} = {v}" for k, v in _leaves(r)][:40]
        parts.append(f"From {m.get('name', 'the tool')}: " + "; ".join(facts) + ".")
    if simulated:
        parts.append("Note: age and gender are simulated.")
    return " ".join(parts)


def _validate(body: dict) -> str | None:
    """Reproduce the request checks the real API makes."""
    if str(body.get("model", "")).startswith("gpt-5") and body.get("temperature", 1) != 1:
        return "Unsupported value: 'temperature' does not support 0 with this model."
    msgs = body.get("messages")
    if not body.get("model") or not isinstance(msgs, list) or not msgs:
        return "model and messages are required"
    for t in body.get("tools", []):
        f = t.get("function", {})
        if t.get("type") != "function" or not f.get("name") or f.get("parameters", {}).get("type") != "object":
            return f"bad tool definition: {t}"

        def walk(s):
            if isinstance(s, dict):
                if s.get("type") == "array" and not s.get("items"):
                    return "array schema missing items"
                if "$ref" in s:
                    return "unresolved $ref"
                return next((e for e in map(walk, s.values()) if e), None)
            if isinstance(s, list):
                return next((e for e in map(walk, s) if e), None)
        if (e := walk(f["parameters"])):
            return f"Invalid schema for function '{f['name']}': {e}"
    pending: set = set()
    for m in msgs:
        role = m.get("role")
        if role == "tool":
            if m.get("tool_call_id") not in pending:
                return "tool message without a preceding tool call"
            pending.discard(m["tool_call_id"])
            continue
        if pending:
            return "assistant tool_calls must be followed by tool messages for every tool_call_id"
        if role == "assistant" and m.get("tool_calls"):
            pending = {c["id"] for c in m["tool_calls"]}
            for c in m["tool_calls"]:
                json.loads(c["function"]["arguments"])
    if pending:
        return "assistant tool_calls must be followed by tool messages"
    return None


class Handler(BaseHTTPRequestHandler):
    requests_seen: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests_seen.append(body)
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "not found"}})
        if self.headers.get("Authorization", "") != "Bearer test-key":
            return self._send(401, {"error": {"message": "bad key"}})
        if (err := _validate(body)):
            return self._send(400, {"error": {"message": err, "type": "invalid_request_error"}})
        msgs = body["messages"]
        message = {"role": "assistant", "content": None}
        if "tools" not in body:  # explainer (or step-limit) call: plain text, no new numbers
            message["content"] = ("In plain words: the tool results show how the groups in this "
                                  "dataset differ; this is educational and not medical advice.")
        elif "keep calling tools" in msgs[[m["role"] for m in msgs].index("user", 1)]["content"] \
                and body.get("tool_choice") != "none":
            message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                                      "function": {"name": "describe_variable",
                                                   "arguments": json.dumps({"variable": "radius"})}}]
        elif body.get("tool_choice") == "none":
            message["content"] = _answer_from_tools([m for m in msgs if m["role"] == "tool"][-1:])
        elif msgs[-1]["role"] == "tool":
            tail = []
            for m in reversed(msgs):
                if m["role"] != "tool":
                    break
                tail.insert(0, m)
            message["content"] = _answer_from_tools(tail)
        elif msgs[-1]["role"] == "user" and msgs[-1]["content"].startswith("The verification step"):
            last = [m for m in msgs if m["role"] == "tool"][-1:]
            message["content"] = _answer_from_tools(last)
        else:
            q = msgs[-1]["content"].lower()
            route = next((r for k, r in ROUTES if k in q), ("dataset_overview", {}))
            calls = route if isinstance(route, list) else [route]
            message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                                      "function": {"name": t, "arguments": json.dumps(a)}}
                                     for t, a in calls]
        self._send(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:8], "object": "chat.completion", "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def start() -> tuple[ThreadingHTTPServer, str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
