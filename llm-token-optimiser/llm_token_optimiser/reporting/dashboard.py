"""
dashboard.py — LLM Token Optimiser Streamlit dashboard.
Run with: streamlit run dashboard.py

Requires: pip install streamlit rouge-score
Optional (for BERTScore): pip install bert-score
"""
from optimiser import compress_prompt, estimate_cost

import streamlit as st

st.set_page_config(page_title="LLM Token Optimiser", layout="centered")
st.title("LLM Token Optimiser — Dashboard")
st.caption("Paste a prompt to see baseline vs. optimised token counts and estimated savings.")

model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "claude-3-5-haiku"])
target_ratio = st.slider("Compression target ratio", 0.1, 0.9, 0.5, 0.05)
prompt = st.text_area("Prompt", height=200)

run_bertscore = st.checkbox(
    "Also compute BERTScore (downloads a ~1.4GB model on first run — slow the first time)"
)

if st.button("Analyse") and prompt.strip():
    try:
        st.session_state.result = compress_prompt(prompt, target_ratio=target_ratio)
        st.session_state.prompt_used = prompt
    except Exception as e:
        st.error(f"Compression failed: {e}")
        st.session_state.pop("result", None)

if "result" in st.session_state:
    result = st.session_state.result
    prompt_used = st.session_state.prompt_used

    try:
        cost = estimate_cost(result.optimised_tokens, model=model)
    except Exception as e:
        st.error(f"Cost estimation failed: {e}")
        cost = None

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline tokens", result.baseline_tokens)
    col2.metric("Optimised tokens", result.optimised_tokens)
    col3.metric("Reduction", f"{result.reduction_pct:.1f}%")

    if cost is not None:
        st.write(f"Estimated cost after optimisation: **${cost:.4f}**")

    st.text_area("Compressed prompt", result.compressed_text, height=200)

    # Semantic fidelity is always computed — not optional.
    st.subheader("Semantic Fidelity")
    st.caption("How closely the compressed text matches the original — "
               "report Section 5.3.")

    fcol1, fcol2 = st.columns(2)

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_f1 = scorer.score(prompt_used, result.compressed_text)["rougeL"].fmeasure
        fcol1.metric("ROUGE-L (F1)", f"{rouge_f1:.3f}")
    except ImportError:
        fcol1.error("ROUGE-L requires `pip install rouge-score`")

    if run_bertscore:
        try:
            with st.spinner("Computing BERTScore (first run downloads the model)..."):
                from bert_score import score as bert_score_fn

                _, _, f1 = bert_score_fn(
                    [result.compressed_text], [prompt_used], lang="en", verbose=False
                )
                fcol2.metric("BERTScore (F1)", f"{float(f1[0]):.3f}")
        except ImportError:
            fcol2.warning("BERTScore requires `pip install bert-score`")
    else:
        fcol2.info("BERTScore not run (checkbox above)")

    st.caption(
        "ROUGE-L checks word/order overlap; BERTScore checks semantic "
        "similarity. Extractive compression naturally scores lower on "
        "ROUGE-L than BERTScore since sentences are dropped/reordered "
        "rather than paraphrased word-for-word."
    )