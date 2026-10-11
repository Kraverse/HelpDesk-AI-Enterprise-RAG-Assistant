import config


def test_nvidia_model_is_not_guessed_by_default():
    assert config.NVIDIA_MODEL == ""
    assert config.NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"


def test_validation_rejects_non_https_provider(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_BASE_URL", "http://example.invalid/v1")
    assert any("HTTPS" in message for message in config.validate_settings())


def test_validation_accepts_default_safe_settings(monkeypatch):
    monkeypatch.setattr(config, "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    assert config.validate_settings() == []


def test_malformed_numeric_setting_uses_default(monkeypatch):
    monkeypatch.setenv("TOP_K", "not-a-number")
    assert config._int_setting("TOP_K", 3, 1, 8) == 3


def test_numeric_setting_is_bounded(monkeypatch):
    monkeypatch.setenv("TOP_K", "999")
    assert config._int_setting("TOP_K", 3, 1, 8) == 8
