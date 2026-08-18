from llm_token_optimiser.context_manager.budget import (
    history_tokens,
    is_over_budget,
    message_tokens,
)
from llm_token_optimiser.context_manager.manager import manage_context
from llm_token_optimiser.context_manager.summariser import extractive_summary


def _make_history(n_turns: int, filler: str = "This is a fairly long message about the weekly project status update. "):
    history = []
    for i in range(n_turns):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": filler * 5 + f" Turn {i}."})
    return history


def test_message_tokens_counts_content():
    m = {"role": "user", "content": "Hello there"}
    assert message_tokens(m) > 0


def test_history_tokens_sums_all_messages():
    history = _make_history(4)
    total = history_tokens(history)
    assert total == sum(message_tokens(m) for m in history)


def test_is_over_budget_true_when_large():
    history = _make_history(30)
    assert is_over_budget(history, max_tokens=200) is True


def test_is_over_budget_false_when_small():
    history = [{"role": "user", "content": "Hi"}]
    assert is_over_budget(history, max_tokens=4000) is False


def test_manage_context_returns_unchanged_when_under_budget():
    history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
    result = manage_context(history, max_tokens=4000, keep_recent_turns=6)
    assert result == history


def test_manage_context_condenses_when_over_budget():
    history = _make_history(30)
    result = manage_context(history, max_tokens=200, keep_recent_turns=6)
    assert len(result) == 7  # 1 summary system message + 6 recent turns
    assert result[0]["role"] == "system"
    assert result[1:] == history[-6:]


def test_manage_context_does_not_condense_short_history_even_over_budget():
    history = _make_history(4)
    result = manage_context(history, max_tokens=1, keep_recent_turns=6)
    assert result == history  # nothing to condense — too few turns


def test_extractive_summary_of_empty_history():
    assert extractive_summary([]) == ""


def test_extractive_summary_produces_shorter_text():
    history = _make_history(10)
    summary = extractive_summary(history, target_ratio=0.3)
    flat_len = sum(len(m["content"]) for m in history)
    assert len(summary) < flat_len
