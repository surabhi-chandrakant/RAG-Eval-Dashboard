"""
app.py — Main entry point.
Page: Home / Q&A Interface
"""
import streamlit as st
from pathlib import Path

import config
from engine import query_with_sources, get_collection_stats

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Config validation ─────────────────────────────────────
errors = config.validate_config()
if errors:
    st.error("**Configuration error:**\n\n" + "\n".join(f"- {e}" for e in errors))
    st.info(
        "Copy `.env.example` → `.env` and fill in your Gemini API key.\n\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )
    st.stop()

# ── Session state ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # list of {role, content, sources}
if "eval_buffer" not in st.session_state:
    st.session_state.eval_buffer = []       # QA pairs for RAGAS

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 KB Assistant")
    st.markdown("---")

    stats = get_collection_stats()
    st.metric("Indexed Chunks", stats["total_chunks"])
    st.metric("Unique Files", len(stats["unique_files"]))

    if stats["unique_files"]:
        with st.expander("📂 Indexed files"):
            for f in stats["unique_files"]:
                st.write(f"• {f}")

    st.markdown("---")
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=config.TOP_K)

    st.markdown("---")
    st.caption("Navigation")
    st.page_link("app.py", label="💬 Q&A", icon="💬")
    st.page_link("pages/1_Document_Manager.py", label="📁 Documents", icon="📁")
    st.page_link("pages/2_RAGAS_Evaluation.py", label="📊 Evaluation", icon="📊")

# ── Main UI ───────────────────────────────────────────────
st.title("💬 Company Knowledge Base Q&A")
st.caption(f"Powered by {config.GROQ_MODEL} · ChromaDB · LlamaIndex")

if stats["total_chunks"] == 0:
    st.warning(
        "No documents indexed yet. "
        "Go to **📁 Document Manager** to upload and index your files."
    )

# Chat history display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['file']}** "
                        f"(page {src['page']}, score: {src['score']})"
                    )
                    st.text(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))

# Input
question = st.chat_input(
    "Ask a question about your documents…",
    disabled=(stats["total_chunks"] == 0),
)

if question:
    # Show user message
    st.session_state.chat_history.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # Run RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            try:
                result = query_with_sources(question, top_k=top_k)
            except Exception as exc:
                st.error(f"Query failed: {exc}")
                st.stop()

        answer = result["answer"]
        sources = result["sources"]

        st.markdown(answer)

        if sources:
            with st.expander(f"📎 {len(sources)} source chunks"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}] {src['file']}** "
                        f"(page {src['page']}, score: {src['score']})"
                    )
                    st.text(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))

        # Save to history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

        # Buffer for evaluation
        st.session_state.eval_buffer.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [s["text"] for s in sources],
            }
        )

# Clear chat button
if st.session_state.chat_history:
    if st.button("🗑️ Clear chat", use_container_width=False):
        st.session_state.chat_history = []
        st.rerun()
