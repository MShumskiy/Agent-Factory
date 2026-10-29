import pytest

from agent_factory.llm.privacy import (
    PrivacyError,
    PrivacyMode,
    check_base_url,
    check_model,
    effective_mode,
    is_ollama_cloud,
)

LOCAL, CLOUD = PrivacyMode.LOCAL, PrivacyMode.CLOUD_OK


@pytest.mark.parametrize("model", ["ollama/gemma3:4b", "ollama/gpt-oss:20b", "ollama/llama3.2"])
def test_local_models_are_allowed(model):
    check_model(model, LOCAL)


@pytest.mark.parametrize("model", [
    "openrouter/qwen/qwen3-235b-a22b-2507",
    "anthropic/claude-sonnet-5.5",
    "gemma3:4b",  # no provider prefix: LLMP would not route it to Ollama
    "ollama/gpt-oss:120b-cloud",
    "ollama/deepseek-v3.1:671b-cloud",
    "ollama/glm-4.6:cloud",
])
def test_remote_and_ollama_cloud_models_are_refused_in_local_mode(model):
    with pytest.raises(PrivacyError):
        check_model(model, LOCAL)
    check_model(model, CLOUD)


def test_is_ollama_cloud():
    assert is_ollama_cloud("ollama/qwen3-coder:480b-cloud")
    assert not is_ollama_cloud("ollama/qwen3:8b")


@pytest.mark.parametrize("url", [
    "http://localhost:8000/v1",
    "http://127.0.0.1:8000/v1",
    "http://192.168.1.69:8000/v1",
    "http://10.0.0.5/v1",
])
def test_local_and_private_addresses_are_allowed(url):
    check_base_url(url, LOCAL)


def test_public_addresses_are_refused_in_local_mode():
    with pytest.raises(PrivacyError, match="not on this machine or a private network"):
        check_base_url("http://8.8.8.8:8000/v1", LOCAL)
    check_base_url("http://8.8.8.8:8000/v1", CLOUD)


def test_sensitive_data_forces_local():
    assert effective_mode(None, CLOUD, sensitive=False) is CLOUD
    assert effective_mode(LOCAL, CLOUD, sensitive=False) is LOCAL
    assert effective_mode(None, LOCAL, sensitive=True) is LOCAL
    with pytest.raises(PrivacyError):
        effective_mode(CLOUD, LOCAL, sensitive=True)
    with pytest.raises(PrivacyError):
        effective_mode(None, CLOUD, sensitive=True)
