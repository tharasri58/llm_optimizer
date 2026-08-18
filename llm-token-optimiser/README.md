# LLM Token Optimiser

A Python middleware library that sits between a calling application and
the OpenAI or Anthropic APIs, and reduces the number of tokens sent on
each request without changing what the model is being asked to do.

## Components

| Module | Responsibility |
|---|---|
| `compression/` | Token Compression Module — TF-IDF sentence scoring, instructional-word protection, embedding-based redundancy removal |
| `context_manager/` | Context Window Manager — token-budget tracking and rolling conversation summarisation |
| `middleware/` | API Middleware Layer — `OptimisedOpenAI` / `OptimisedAnthropic` drop-in client wrappers |
| `reporting/` | Dashboard/CLI Tool — token counting, cost estimation, Click CLI, Streamlit dashboard |

## Install

```bash
pip install -e .
# optional extras:
pip install -e ".[dashboard]"   # Streamlit dashboard
pip install -e ".[embeddings]"  # Sentence-BERT redundancy removal (heavier)
pip install -e ".[sdks]"        # openai / anthropic clients
pip install -e ".[dev]"         # pytest, rouge-score, bert-score
```

The core package only requires scikit-learn, nltk, tiktoken, and click.
Sentence-transformers, streamlit, and the provider SDKs are optional —
the code falls back gracefully (TF-IDF-based similarity, heuristic
token counting) when they aren't installed, and openai/anthropic are
only imported when you don't pass in your own `client=`.

## Quick start

```python
from llm_token_optimiser import OptimisedOpenAI

client = OptimisedOpenAI(api_key="...", compression_target=0.5)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": long_user_prompt},
    ],
)

print(client.last_savings_report())
# {'baseline_tokens': 1840, 'optimised_tokens': 760, 'reduction': '58.7%'}
```

## Adaptive local-model support

The package also supports adaptive clients via `OptimisedLLM`.
You can pass any client with an OpenAI-style
`client.chat.completions.create(...)` interface or an Anthropic-style
`client.messages.create(...)` interface.

```python
from llm_token_optimiser import OptimisedLLM

# A local model adapter only needs to expose one of the supported shapes.
client = OptimisedLLM(client=local_model_adapter, compression_target=0.5)
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Summarise this text..."}],
)
print(client.last_savings_report())
```

### Ready-to-run local adapter example

A complete example is available in `examples/llama_cpp_adapter.py`.
It uses `llama-cpp-python` and a local LLaMA-compatible model file.

If `llama-cpp-python` is difficult to install on Windows, you can instead use
`examples/transformers_adapter.py` with a Hugging Face Transformers model.

Install the local runtime optional dependency for transformers:

```bash
pip install -e ".[transformers]"
```

Run the transformers example:

```bash
python examples/transformers_adapter.py \
  --model-name distilgpt2 \
  --prompt-file examples/prompts/example.txt
```

The default `distilgpt2` model works without additional conversion steps.

For a full example and prompt file, see `examples/README.md` and
`examples/prompts/example.txt`.

## CLI

```bash
token-optimiser analyse prompt.txt
```

## Dashboard

```bash
streamlit run llm_token_optimiser/reporting/dashboard.py
```

## Local Transformers dashboard

Install the dashboard and Transformers extras:

```bash
pip install -e .
pip install -e "[dashboard,transformers]"
```

If Streamlit still fails due to image-processing imports, also install `torchvision`:

```bash
pip install torchvision
```

Run the local LLM app:

```bash
streamlit run llm_token_optimiser/reporting/dashboard_local.py
```

## Tests

```bash
pytest tests/ -v
```

## License

For academic submission purposes.
