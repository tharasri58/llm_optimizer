"""Example adapter for running a local Transformers model with OptimisedLLM.

This example uses the Hugging Face Transformers library and a compatible
causal language model. It adapts the model into an OpenAI-style client
shape so `OptimisedLLM` can compress prompts before generation.

Usage:
    pip install -e .
    pip install -e ".[transformers]"

    python examples/transformers_adapter.py \
      --model-name distilgpt2 \
      --prompt-file examples/prompts/example.txt
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_token_optimiser import OptimisedLLM


class TransformersOpenAIAdapter:
    """Adapter exposing an OpenAI-style chat completions interface."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.chat = self
        self.completions = self

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 128,
        temperature: float = 1.0,
        **generate_kwargs,
    ):  # pragma: no cover
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        prompt_lines: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            label = role_map.get(role, role.capitalize())
            prompt_lines.append(f"{label}: {content}")
        prompt_lines.append("Assistant:")
        return "\n".join(prompt_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Transformers model through OptimisedLLM.")
    parser.add_argument("--model-name", default="distilgpt2", help="Hugging Face model name")
    parser.add_argument("--prompt-file", required=True, help="Text file containing the user prompt.")
    parser.add_argument("--compression-target", type=float, default=0.5, help="Compression target ratio.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens for the model response.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    args = parser.parse_args()

    prompt_text = open(args.prompt_file, encoding="utf-8").read()
    local_client = TransformersOpenAIAdapter(model_name=args.model_name)
    client = OptimisedLLM(client=local_client, compression_target=args.compression_target)

    response = client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": prompt_text}],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("--- MODEL RESPONSE ---")
    print(response["choices"][0]["message"]["content"])
    print("--- SAVINGS REPORT ---")
    print(client.last_savings_report())


if __name__ == "__main__":
    main()
