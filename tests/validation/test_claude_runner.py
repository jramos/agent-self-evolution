"""ClaudeCodeAgentRunner stream-json parsing (no live `claude` calls)."""
from evolution.validation.claude_runner import _parse_stream_json

STREAM = "\n".join([
    '{"type":"system","subtype":"init","model":"claude-sonnet-4-6","tools":["Bash"]}',
    '{"type":"assistant","message":{"content":[{"type":"text","text":"thinking"},'
    '{"type":"tool_use","name":"Bash","input":{"command":"./bin/check"}}]}}',
    '{"type":"result","subtype":"success","is_error":false,"result":"done",'
    '"total_cost_usd":0.012,"usage":{"input_tokens":10,"output_tokens":20,'
    '"cache_read_input_tokens":5,"cache_creation_input_tokens":3}}',
])


def test_parse_extracts_tool_calls_cost_tokens():
    r = _parse_stream_json(STREAM, duration_seconds=2.0)
    assert r.error is None
    assert r.tool_calls_seq == ["Bash"]
    assert r.tool_calls_with_args == [{"name": "Bash", "arguments": {"command": "./bin/check"}}]
    assert r.agent_cost_usd == 0.012 and r.agent_cost_source == "actual"
    assert r.agent_tokens["input_tokens"] == 10 and r.agent_tokens["output_tokens"] == 20
    assert r.model_name == "claude-sonnet-4-6"
    assert r.final_text_tail == "done"


def test_parse_flags_error_result():
    stream = ('{"type":"result","subtype":"success","is_error":true,'
              '"result":"Not logged in","total_cost_usd":0,"usage":{}}')
    r = _parse_stream_json(stream, duration_seconds=1.0)
    assert r.error is not None and "Not logged in" in r.error


def test_parse_no_result_event_is_error():
    r = _parse_stream_json('{"type":"system","subtype":"init","model":"m"}', duration_seconds=1.0)
    assert r.error is not None


def test_parse_zero_cost_is_uncaptured_not_actual():
    stream = ('{"type":"result","subtype":"success","is_error":false,"result":"ok",'
              '"total_cost_usd":0,"usage":{"input_tokens":1,"output_tokens":1}}')
    r = _parse_stream_json(stream, duration_seconds=1.0)
    assert r.agent_cost_usd is None and r.agent_cost_source == "uncaptured"
