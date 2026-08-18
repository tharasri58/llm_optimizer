"""
dashboard.py — Streamlit dashboard for the Dashboard/CLI Tool.

Streamlit is an optional dependency: this module only imports it inside
``run()``, so importing the package (or running the CLI/tests) never
requires streamlit to be installed. Launch with:

    streamlit run -m llm_token_optimiser.reporting.dashboard
"""
from __future__ import annotations

from llm_token_optimiser.compression.compressor import compress_prompt
from llm_token_optimiser.reporting.token_counter import estimate_cost


def run():
    import streamlit as st

    st.set_page_config(page_title="LLM Token Optimiser", layout="centered")
    st.title("LLM Token Optimiser — Dashboard")
    st.caption("Paste a prompt to see baseline vs. optimised token counts and estimated savings.")

    model = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "claude-3-5-haiku"]
    )
    target_ratio = st.slider("Compression target ratio", 0.1, 0.9, 0.5, 0.05)
    prompt = st.text_area("Prompt", height=200)

    if st.button("Analyse") and prompt.strip():
        result = compress_prompt(prompt, target_ratio=target_ratio, model=model)
        cost = estimate_cost(result.optimised_tokens, model=model)

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline tokens", result.baseline_tokens)
        col2.metric("Optimised tokens", result.optimised_tokens)
        col3.metric("Reduction", f"{result.reduction_pct}%")

        st.write(f"Estimated cost after optimisation: **${cost}**")
        st.text_area("Compressed prompt", result.compressed_text, height=200)


if __name__ == "__main__":
    run()
