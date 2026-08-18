# Local Example: llama_cpp_adapter

This example shows how to run a local LLaMA-compatible model with `OptimisedLLM`.

## Setup

```bash
pip install -e .
pip install -e ".[local]"
```

## Run

```bash
python examples/llama_cpp_adapter.py \
  --model-path /path/to/model.bin \
  --prompt-file examples/prompts/example.txt
```

## Transformers alternative

If `llama-cpp-python` is difficult to install on Windows, use the Transformers example instead:

```bash
pip install -e .
pip install -e ".[transformers]"
python examples/transformers_adapter.py \
  --model-name distilgpt2 \
  --prompt-file examples/prompts/example.txt
```
## Streamlit local LLM app

If you want a Streamlit interface for a local Transformers model, install the dashboard and transformers extras:

```bash
pip install -e .
pip install -e "[dashboard,transformers]"
```

Then run:

```bash
streamlit run llm_token_optimiser/reporting/dashboard_local.py
```
## Notes

- `model.bin` must be compatible with `llama-cpp-python`.
- The example uses the OpenAI-style client shape for `OptimisedLLM`.
- You can change the prompt by editing `examples/prompts/example.txt`.
