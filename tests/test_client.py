import json

import httpx
import pytest

from agent_factory.llm.client import LLMError, LLMPClient, LLMPSettings, inline_schema
from agent_factory.llm.privacy import PrivacyError, PrivacyMode
from agent_factory.sim.models import Action, Resolution

ACTION = {"intent": "hold", "orders": ["dig in"], "rationale": "safe"}


def completion(content: str) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content}}],
    })


class Server:
    """Records requests and answers them from a list of replies."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return completion(self.replies.pop(0))


def client(server, privacy=PrivacyMode.LOCAL, base="http://127.0.0.1:8000/v1", cache_dir=None) -> LLMPClient:
    http = httpx.Client(transport=httpx.MockTransport(server))
    return LLMPClient(LLMPSettings(base, "key"), privacy, "run1", cache_dir=cache_dir, http_client=http)


def ask(llm: LLMPClient, schema=Action, model="ollama/gemma3:4b"):
    return llm.structured(schema, model=model, system="s", user="u", agent="actor.blue",
                          temperature=0.5, seed=42)


def test_request_carries_schema_seed_and_llmp_tags():
    server = Server(json.dumps(ACTION))
    assert ask(client(server)).intent == "hold"
    body = server.requests[0]
    assert body["model"] == "ollama/gemma3:4b" and body["seed"] == 42
    assert body["response_format"]["type"] == "json_schema"
    assert body["llmp"] == {"src": "agent_factory", "run_id": "run1", "agent": "actor.blue",
                            "session_id": "run1/actor.blue", "log": "meta"}


def test_cloud_ok_leaves_llmp_logging_at_its_default():
    server = Server(json.dumps(ACTION))
    ask(client(server, privacy=PrivacyMode.CLOUD_OK), model="openrouter/qwen/qwen3-235b-a22b-2507")
    assert "log" not in server.requests[0]["llmp"]


def test_privacy_guard_runs_before_any_request():
    server = Server(json.dumps(ACTION))
    with pytest.raises(PrivacyError):
        ask(client(server), model="openrouter/qwen/qwen3-235b-a22b-2507")
    with pytest.raises(PrivacyError):
        ask(client(server), model="ollama/gpt-oss:120b-cloud")
    assert server.requests == []
    with pytest.raises(PrivacyError):
        client(server, base="http://8.8.8.8:8000/v1")


def test_invalid_reply_gets_one_repair_attempt():
    server = Server("Sure! Here is my move: hold the line.", "```json\n" + json.dumps(ACTION) + "\n```")
    assert ask(client(server)).orders == ["dig in"]
    assert len(server.requests) == 2
    repair = server.requests[1]["messages"]
    assert repair[-2]["role"] == "assistant" and "does not match" in repair[-1]["content"]


def test_two_invalid_replies_raise():
    server = Server("no", "still no")
    with pytest.raises(LLMError, match="did not return valid Action"):
        ask(client(server))


def test_cache_replays_identical_requests(tmp_path):
    server = Server(json.dumps(ACTION))
    llm = client(server, cache_dir=tmp_path)
    first, second = ask(llm), ask(llm)
    assert first == second
    assert len(server.requests) == 1 and llm.requests == 1 and llm.cache_hits == 1


def test_connection_errors_name_llmp():
    def refuse(request):
        raise httpx.ConnectError("refused")

    llm = client(refuse)
    llm._client = llm._client.with_options(max_retries=0)
    with pytest.raises(LLMError, match="cannot reach LLMP"):
        ask(llm)


def test_inline_schema_has_no_refs():
    schema = json.dumps(inline_schema(Resolution))
    assert "$ref" not in schema and "$defs" not in schema
    assert '"path"' in schema  # the nested StateChange fields are inlined
