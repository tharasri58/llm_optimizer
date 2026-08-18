from llm_token_optimiser.middleware.anthropic_wrapper import OptimisedAnthropic
from llm_token_optimiser.middleware.llm_wrapper import OptimisedLLM
from llm_token_optimiser.middleware.openai_wrapper import OptimisedOpenAI


# ---- Mock SDK clients -------------------------------------------------
# These stand in for `openai.OpenAI()` / `anthropic.Anthropic()` so the
# wrapper's compression/routing logic can be tested without a live API
# key. They just record what was sent and echo a canned response.

class _MockOpenAIChatCompletions:
    def __init__(self):
        self.last_call = None

    def create(self, model, messages, **kwargs):
        self.last_call = {"model": model, "messages": messages, **kwargs}
        return {"choices": [{"message": {"role": "assistant", "content": "mock reply"}}]}


class _MockOpenAIChat:
    def __init__(self):
        self.completions = _MockOpenAIChatCompletions()


class MockOpenAIClient:
    def __init__(self):
        self.chat = _MockOpenAIChat()


class _MockAnthropicMessages:
    def __init__(self):
        self.last_call = None

    def create(self, model, messages, max_tokens, **kwargs):
        self.last_call = {"model": model, "messages": messages, "max_tokens": max_tokens, **kwargs}
        return {"content": [{"type": "text", "text": "mock reply"}]}


class MockAnthropicClient:
    def __init__(self):
        self.messages = _MockAnthropicMessages()


# ---- OpenAI wrapper tests ----------------------------------------------

def test_openai_wrapper_single_prompt_is_compressed():
    mock_client = MockOpenAIClient()
    client = OptimisedOpenAI(client=mock_client, compression_target=0.5)

    long_prompt = (
        "This is a long, repetitive prompt. This is a long, repetitive prompt. "
        "It says the same thing more than once, on purpose, for this test."
    )
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": long_prompt}]
    )

    sent = mock_client.chat.completions.last_call["messages"][0]["content"]
    assert len(sent) <= len(long_prompt)
    report = client.last_savings_report()
    assert "baseline_tokens" in report and "optimised_tokens" in report


def test_openai_wrapper_passes_through_tool_calls_unmodified():
    mock_client = MockOpenAIClient()
    client = OptimisedOpenAI(client=mock_client)

    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "Sunny, 25C"},
    ]
    client.chat.completions.create(model="gpt-4o-mini", messages=messages)

    sent = mock_client.chat.completions.last_call["messages"]
    assert sent == messages  # untouched, per Section 4.4


def test_openai_wrapper_returns_underlying_response():
    mock_client = MockOpenAIClient()
    client = OptimisedOpenAI(client=mock_client)
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}]
    )
    assert response["choices"][0]["message"]["content"] == "mock reply"


# ---- Anthropic wrapper tests --------------------------------------------

def test_anthropic_wrapper_single_prompt_is_compressed():
    mock_client = MockAnthropicClient()
    client = OptimisedAnthropic(client=mock_client, compression_target=0.5)

    long_prompt = (
        "This is a long, repetitive prompt. This is a long, repetitive prompt. "
        "It says the same thing more than once, on purpose, for this test."
    )
    client.messages.create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": long_prompt}],
        max_tokens=200,
    )

    sent = mock_client.messages.last_call["messages"][0]["content"]
    assert len(sent) <= len(long_prompt)


def test_anthropic_wrapper_returns_underlying_response():
    mock_client = MockAnthropicClient()
    client = OptimisedAnthropic(client=mock_client)
    response = client.messages.create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    assert response["content"][0]["text"] == "mock reply"


def test_anthropic_wrapper_multi_turn_routes_through_context_manager():
    mock_client = MockAnthropicClient()
    client = OptimisedAnthropic(client=mock_client, max_context_tokens=50)

    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "Filler turn about the project. " * 10}
        for i in range(20)
    ]
    client.messages.create(model="claude-3-5-sonnet", messages=history, max_tokens=100)

    sent = mock_client.messages.last_call["messages"]
    assert len(sent) < len(history)


# ---- Adaptive wrapper tests -------------------------------------------

def test_optimised_llm_detects_openai_style_client():
    mock_client = MockOpenAIClient()
    client = OptimisedLLM(client=mock_client)

    client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert mock_client.chat.completions.last_call["model"] == "gpt-4o-mini"


def test_optimised_llm_detects_anthropic_style_client():
    mock_client = MockAnthropicClient()
    client = OptimisedLLM(client=mock_client)

    client.messages.create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    assert mock_client.messages.last_call["model"] == "claude-3-5-sonnet"
