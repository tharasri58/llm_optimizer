"""Streamlit app for running a local Transformers LLM through OptimisedLLM."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from llm_token_optimiser.compression.compressor import compress_prompt
from llm_token_optimiser.reporting.token_counter import estimate_cost
from llm_token_optimiser.middleware.llm_wrapper import OptimisedLLM

try:
    from llm_token_optimiser.middleware.transformers_wrapper import TransformersOpenAIAdapter
except ImportError:  # pragma: no cover
    TransformersOpenAIAdapter = None  # type: ignore


def run() -> None:
    st.set_page_config(page_title="LLM Token Optimiser — Local LLM", layout="centered")
    st.title("Local LLM Token Optimiser")
    st.caption("Run a local Transformers model with prompt compression and savings reporting.")

    model_name = st.text_input("Transformers model name", value="distilgpt2")
    compression_target = st.slider("Compression target ratio", 0.1, 0.9, 0.5, 0.05)
    max_tokens = st.number_input("Max generation tokens", min_value=16, max_value=512, value=128, step=16)
    temperature = st.slider("Temperature", 0.1, 1.5, 1.0, 0.1)
    prompt = st.text_area("Prompt", height=240)

    if TransformersOpenAIAdapter is None:
        st.error(
            "The local Transformers adapter requires optional dependencies. "
            "Install with `pip install -e \".[transformers]\"` and restart Streamlit."
        )
        return

    if st.button("Run local model") and prompt.strip():
        with st.spinner("Loading model and generating response..."):
            local_client = TransformersOpenAIAdapter(model_name=model_name)
            client = OptimisedLLM(client=local_client, compression_target=compression_target)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        result = compress_prompt(prompt, target_ratio=compression_target, model=model_name)
        cost = estimate_cost(result.optimised_tokens, model=model_name)

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline tokens", result.baseline_tokens)
        col2.metric("Optimised tokens", result.optimised_tokens)
        col3.metric("Reduction", f"{result.reduction_pct}%")

        st.write(f"Estimated cost after optimisation: **${cost}**")
        st.subheader("Model response")
        st.write(response["choices"][0]["message"]["content"])
        st.subheader("Compressed prompt")
        st.text_area("Compressed prompt", result.compressed_text, height=200)


if __name__ == "__main__":
    run()
