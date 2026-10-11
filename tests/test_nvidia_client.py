import io
import json
from urllib.error import HTTPError

import pytest

from utils import nvidia_client as client


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def read(self):
        return self.body


def test_model_must_be_explicit(monkeypatch):
    monkeypatch.setattr(client, "NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(client, "NVIDIA_MODEL", "")
    with pytest.raises(client.ProviderError, match="model is not configured"):
        client.chat_completion([{"role": "user", "content": "hello"}])


def test_request_uses_configured_model_and_bearer_key(monkeypatch):
    monkeypatch.setattr(client, "NVIDIA_API_KEY", "test-secret")
    monkeypatch.setattr(client, "NVIDIA_MODEL", "approved/model-id")
    monkeypatch.setattr(client, "NVIDIA_MAX_RETRIES", 0)
    captured = {}
    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return FakeResponse({"choices": [{"message": {"content": "Grounded answer [S1]"}}]})
    monkeypatch.setattr(client, "urlopen", fake_urlopen)
    result = client.chat_completion([{"role": "user", "content": "question"}])
    assert result.text == "Grounded answer [S1]"
    assert result.model == "approved/model-id"
    assert captured["auth"] == "Bearer test-secret"
    assert captured["payload"]["model"] == "approved/model-id"
    assert captured["url"].endswith("/chat/completions")
    assert captured["timeout"] > 0


def test_http_errors_are_sanitized(monkeypatch):
    monkeypatch.setattr(client, "NVIDIA_API_KEY", "never-log-this")
    monkeypatch.setattr(client, "NVIDIA_MODEL", "approved/model-id")
    monkeypatch.setattr(client, "NVIDIA_MAX_RETRIES", 0)
    def fail(*_args, **_kwargs):
        raise HTTPError("https://example.invalid", 401, "Unauthorized", {}, io.BytesIO(b"never-log-this"))
    monkeypatch.setattr(client, "urlopen", fail)
    with pytest.raises(client.ProviderError) as error:
        client.chat_completion([{"role": "user", "content": "question"}])
    assert "authentication" in str(error.value)
    assert "never-log-this" not in str(error.value)


def test_invalid_json_is_sanitized(monkeypatch):
    monkeypatch.setattr(client, "NVIDIA_API_KEY", "secret")
    monkeypatch.setattr(client, "NVIDIA_MODEL", "approved/model-id")
    monkeypatch.setattr(client, "NVIDIA_MAX_RETRIES", 0)
    class Invalid(FakeResponse):
        def read(self): return b"not-json"
    monkeypatch.setattr(client, "urlopen", lambda *_a, **_k: Invalid({}))
    with pytest.raises(client.ProviderError, match="invalid response"):
        client.chat_completion([{"role": "user", "content": "question"}])
