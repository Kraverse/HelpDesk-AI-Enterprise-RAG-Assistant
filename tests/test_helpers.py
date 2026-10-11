from types import SimpleNamespace

from utils.helpers import chat_history_to_string, format_sources


def test_sources_are_deduplicated_in_order():
    docs = [
        SimpleNamespace(metadata={"source": "faq.txt"}),
        SimpleNamespace(metadata={"source": "faq.txt"}),
        SimpleNamespace(metadata={"source": "returns.txt"}),
    ]
    assert format_sources(docs) == ["faq.txt", "returns.txt"]


def test_chat_export_includes_source_citations():
    exported = chat_history_to_string([
        {
            "role": "assistant",
            "content": "Business hours are weekdays [S1].",
            "citations": [{"id": "S1", "source": "faq.txt"}],
        }
    ])
    assert "Business hours are weekdays [S1]." in exported
    assert "Sources: faq.txt" in exported
