import sys
from types import SimpleNamespace

import config


def test_openrouter_default_configuration():
    assert config.OPENROUTER_MODEL == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert config.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_validation_rejects_non_https_provider(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_BASE_URL", "http://example.invalid/v1")
    assert any("HTTPS" in message for message in config.validate_settings())


def test_validation_accepts_default_safe_settings(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    assert config.validate_settings() == []


def test_openrouter_environment_variable_is_secret_fallback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-test-secret")
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=SimpleNamespace(get=lambda _name: None)))
    assert config._secret("OPENROUTER_API_KEY") == "local-test-secret"


def test_streamlit_secret_takes_precedence_over_local_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "local-fallback")
    secrets = SimpleNamespace(get=lambda _name: "streamlit-secret")
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=secrets))
    assert config._secret("OPENROUTER_API_KEY") == "streamlit-secret"


def test_openrouter_model_supports_streamlit_secret_and_environment_fallback(monkeypatch):
    default = "nvidia/nemotron-3-ultra-550b-a55b:free"
    monkeypatch.setenv("OPENROUTER_MODEL", "local/model")
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=SimpleNamespace(get=lambda _name: None)))
    assert config._secret("OPENROUTER_MODEL", default) == "local/model"
    monkeypatch.setitem(sys.modules, "streamlit", SimpleNamespace(secrets=SimpleNamespace(get=lambda _name: "secret/model")))
    assert config._secret("OPENROUTER_MODEL", default) == "secret/model"


def test_malformed_numeric_setting_uses_default(monkeypatch):
    monkeypatch.setenv("TOP_K", "not-a-number")
    assert config._int_setting("TOP_K", 3, 1, 8) == 3


def test_upload_limits_are_bounded():
    assert config.MAX_UPLOAD_FILE_BYTES == 10 * 1024 * 1024
    assert config.MAX_UPLOAD_DOCUMENTS == 20


def test_numeric_setting_is_bounded(monkeypatch):
    monkeypatch.setenv("TOP_K", "999")
    assert config._int_setting("TOP_K", 3, 1, 8) == 8
