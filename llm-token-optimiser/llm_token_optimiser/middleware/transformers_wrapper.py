"""Adapter for local Transformers models exposed as an OpenAI-style client."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersOpenAIAdapter:
    """Wrap a Transformers causal LM as an OpenAI-style chat client."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        if self.model.config.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.chat = self
        self.completions = self

    def create(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 128,
        temperature: float = 1.0,
        **kwargs,
    ):
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=self.model.config.pad_token_id,
            **kwargs,
        )
        generated = output[0][ inputs["input_ids"].shape[-1] : ]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}

    def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        lines: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            label = role_map.get(role, role.capitalize())
            lines.append(f"{label}: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)
