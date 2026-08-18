"""Example adapter for running a local LLaMA-compatible model with OptimisedLLM.

This file is a ready-to-run example that adapts a local `llama-cpp-python`
runtime into an OpenAI-style client shape.

Usage:
    pip install -e .
    pip install llama-cpp-python

    python examples/llama_cpp_adapter.py \
        --model-path C:/path/to/model.bin \
        --prompt-file prompts/example.txt

If you do not have `llama-cpp-python`, install it with:
    pip install llama-cpp-python

If you want to use a Hugging Face-style model, convert the weights to a
compatible `.bin` file for llama-cpp.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_cpp import Llama
from llm_token_optimiser import OptimisedLLM


class LlamaCppOpenAIAdapter:
    """Adapter exposing OpenAI-style chat completions for a local Llama model."""

    def __init__(self, model_path: str, temperature: float = 0.7):
        self.llm = Llama(model_path=model_path, temperature=temperature)
        self.temperature = temperature
        self.chat = self
        self.completions = self

    def create(self, model: str, messages: List[Dict[str, Any]], max_tokens: int = 256, temperature: Optional[float] = None, **kwargs):
        prompt = self._build_prompt(messages)
        response = self.llm.create(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        )
        text = response["choices"][0]["text"]
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        prompt_parts: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            label = role_map.get(role, role.capitalize())
            prompt_parts.append(f"{label}: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local llama-cpp model through OptimisedLLM.")
    parser.add_argument("--model-path", required=True, help="Path to a llama-cpp-compatible model file.")
    parser.add_argument("--prompt-file", required=True, help="Text file containing the user prompt.")
    parser.add_argument("--compression-target", type=float, default=0.5, help="Compression target ratio.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens for the local model response.")
    args = parser.parse_args()

    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    local_client = LlamaCppOpenAIAdapter(model_path=args.model_path)
    client = OptimisedLLM(client=local_client, compression_target=args.compression_target)

    response = client.chat.completions.create(
        model="local-llama",
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=args.max_tokens,
    )

    print("--- MODEL RESPONSE ---")
    print(response["choices"][0]["message"]["content"])
    print("--- SAVINGS REPORT ---")
    print(client.last_savings_report())


if __name__ == "__main__":
    main()
