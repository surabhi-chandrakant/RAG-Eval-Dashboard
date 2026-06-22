"""RAGAS Evaluation Dashboard."""
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import config
from evaluator import RAGAS_AVAILABLE, RAGAS_ERROR, interpret_score, run_evaluation, score_color

st.set_page_config(page_title="RAGAS Evaluation", page_icon="📊", layout="wide")

errors = config.validate_config()
if errors:
    st.error("\n".join(errors))
    st.stop()

if "eval_buffer" not in st.session_state:
    st.session_state.eval_buffer = []
if "eval_results_df" not in st.session_state:
    st.session_state.eval_results_df = None
if "eval_agg" not in st.session_state:
    st.session_state.eval_agg = None

with st.sidebar:
    st.title("📊 RAGAS Evaluation")
    buf_count = len(st.session_state.eval_buffer)
    st.metric("Buffered Q&A pairs", buf_count)

st.title("📊 RAGAS Evaluation Dashboard")
st.caption("Evaluate RAG quality — faithfulness and context utilization. No labels needed.")

if not RAGAS_AVAILABLE:
    st.error("RAGAS import failed.")
    if RAGAS_ERROR:
        st.code("Error: " + RAGAS_ERROR)
    st.info("Run in your terminal: pip install ragas==0.1.21 langchain-groq==0.1.10 langchain-community==0.2.19")
    st.stop()

with st.expander("ℹ️ Metrics explained"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Faithfulness** — Are all claims in the answer supported by the retrieved context? Low score = hallucination risk.")
    with c2:
        st.markdown("**Context Utilization** — Did the answer actually make use of the retrieved context? Low score = context being ignored.")

st.divider()
tab1, tab2 = st.tabs(["🧪 From session", "📥 Manual JSON"])

with tab1:
    if st.session_state.eval_buffer:
        n = len(st.session_state.eval_buffer)
        st.info(f"**{n} Q&A pair(s)** buffered from your chat session.")
        with st.expander("Preview pairs"):
            for i, p in enumerate(st.session_state.eval_buffer, 1):
                question = p["question"]
                answer = p["answer"][:200]
                st.markdown(f"**Q{i}:** {question}")
                st.markdown(f"**A{i}:** {answer}…")
                st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
                with st.spinner("Running RAGAS (~20-40s per question via Groq)…"):
                    try:
                        df, agg = run_evaluation(st.session_state.eval_buffer)
                        st.session_state.eval_results_df = df
                        st.session_state.eval_agg = agg
                        st.success("Evaluation complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        with c2:
            if st.button("🗑️ Clear buffer", use_container_width=True):
                st.session_state.eval_buffer = []
                st.rerun()
    else:
        st.info("Ask questions in 💬 Q&A first, then return here to evaluate.")

with tab2:
    example = json.dumps([{
        "question": "What is the refund policy?",
        "answer": "Refunds are allowed within 30 days of purchase.",
        "contexts": ["Our policy allows full refunds within 30 days of purchase."]
    }], indent=2)
    manual = st.text_area("JSON array of QA pairs", value=example, height=220)
    if st.button("▶️ Evaluate Manual Input", type="primary"):
        try:
            pairs = json.loads(manual)
            if not isinstance(pairs, list):
                st.error("Input must be a JSON array.")
            else:
                with st.spinner("Running RAGAS…"):
                    df, agg = run_evaluation(pairs)
                    st.session_state.eval_results_df = df
                    st.session_state.eval_agg = agg
                    st.success("Done!")
                    st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        except Exception as e:
            st.error(f"Failed: {e}")

st.divider()

if st.session_state.eval_results_df is not None:
    df = st.session_state.eval_results_df
    agg = st.session_state.eval_agg
    st.subheader("📈 Results")

    cols = st.columns(len(agg))
    for col, (metric, score) in zip(cols, agg.items()):
        with col:
            color = score_color(score)
            label = metric.replace("_", " ").title()
            interp = interpret_score(metric, score)
            st.markdown(
                f'<div style="background:{color}22;border:1px solid {color};'
                f'border-radius:8px;padding:16px;text-align:center;">'
                f'<div style="font-size:2rem;font-weight:bold;color:{color}">{score:.2f}</div>'
                f'<div style="font-weight:600">{label}</div>'
                f'<div style="font-size:0.8rem;color:#888">{interp}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    ml = list(agg.keys())
    sl = list(agg.values())
    theta = [m.replace("_", " ").title() for m in ml]
    fig = go.Figure(go.Scatterpolar(
        r=sl + [sl[0]],
        theta=theta + [theta[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="rgba(99,102,241,0.9)", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title="Aggregate RAGAS Scores",
        height=350, margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Results Table"):
        st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ragas_results.csv",
        mime="text/csv",
    )