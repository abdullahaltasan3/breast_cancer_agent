"""Shows the mock server really enforces the API rules, so passing e2e tests mean something."""
from mock_openai_server import _validate

GOOD_TOOL = {"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {}}}}


def test_array_without_items_rejected():
    bad = {"type": "function", "function": {"name": "t", "parameters": {
        "type": "object", "properties": {"x": {"type": "array"}}}}}
    assert "missing items" in _validate({"model": "m", "messages": [{"role": "user", "content": "q"}], "tools": [bad]})


def test_tool_call_without_tool_message_rejected():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function",
                                                                    "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "user", "content": "next"}]
    assert _validate({"model": "m", "messages": msgs, "tools": [GOOD_TOOL]})


def test_valid_request_accepted():
    assert _validate({"model": "m", "messages": [{"role": "user", "content": "q"}], "tools": [GOOD_TOOL]}) is None
