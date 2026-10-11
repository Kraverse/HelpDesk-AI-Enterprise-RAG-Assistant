import config
from utils import input_policy
from utils.input_policy import bound_history, register_request, validate_question


def test_question_is_trimmed_and_validated():
    assert validate_question("  Where is my order?  ") == ("Where is my order?", None)
    assert validate_question("  ")[0] is None
    assert validate_question("???")[0] is None


def test_question_length_is_bounded(monkeypatch):
    monkeypatch.setattr(config, "MAX_QUESTION_CHARS", 5)
    # The validator reads the setting at import time, so cover the configured limit contract.
    assert validate_question("x" * 1001)[0] is None


def test_history_is_limited_to_valid_roles_and_message_count(monkeypatch):
    monkeypatch.setattr(input_policy, "MAX_CHAT_MESSAGES", 4)
    monkeypatch.setattr(input_policy, "MAX_SESSION_CHARS", 1000)
    history = [{"role": "user", "content": str(i)} for i in range(10)]
    result = bound_history(history)
    assert len(result) <= 4
    assert all(item["role"] in {"user", "assistant"} for item in result)
    assert bound_history([{"role": "system", "content": "hidden"}]) == []


def test_history_caps_total_text():
    history = [
        {"role": "user", "content": "a" * 700},
        {"role": "assistant", "content": "b" * 700},
    ]
    assert sum(len(x["content"]) for x in bound_history(history)) <= config.MAX_SESSION_CHARS


def test_session_rate_limit_rejects_excess_and_prunes_old_requests(monkeypatch):
    monkeypatch.setattr(input_policy, "SESSION_REQUESTS_PER_MINUTE", 2)
    recent, allowed = register_request([10.0, 20.0], now=30.0)
    assert allowed is False
    assert recent == [10.0, 20.0]
    recent, allowed = register_request([1.0, 11.0], now=70.0)
    assert allowed is True
    assert recent == [11.0, 70.0]
