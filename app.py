"""
app.py — Main Q&A page.

Fixes:
  1. Input validation — rejects blank/special-character-only queries
  2. Rate limit handling — catches 429 and shows retry countdown
"""
import re
import time
import streamlit as st

import config
from engine import query_with_sources, get_collection_stats

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
    st.stop()

# ── Session state ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_buffer" not in st.session_state:
    st.session_state.eval_buffer = []

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

# ── Helpers ───────────────────────────────────────────────

def _is_valid_question(text: str) -> tuple[bool, str]:
    """
    Returns (is_valid, error_message).
    Rejects: empty, whitespace-only, special-character-only inputs.
    """
    stripped = text.strip()
    if not stripped:
        return False, "Please enter a question."
    # Must contain at least one letter or digit
    if not re.search(r"[a-zA-Z0-9]", stripped):
        return False, "Please enter a question with actual words or numbers."
    if len(stripped) < 3:
        return False, "Question is too short. Please be more specific."
    return True, ""


def _parse_retry_seconds(error_str: str) -> int:
    """Extract retry delay from a 429 error message, default 60s."""
    match = re.search(r"retry[^\d]*(\d+)", str(error_str), re.IGNORECASE)
    return int(match.group(1)) if match else 60


# ── Main UI ───────────────────────────────────────────────
st.title("💬 Company Knowledge Base Q&A")
st.caption(f"Powered by {config.GROQ_MODEL} · ChromaDB · LlamaIndex")

if stats["total_chunks"] == 0:
    st.warning(
        "No documents indexed yet. "
        "Go to **📁 Document Manager** to upload and index your files."
    )

# Chat history
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
    # ── Bug fix 1: validate input before sending to LLM ──
    valid, validation_msg = _is_valid_question(question)
    if not valid:
        st.warning(validation_msg)
        st.stop()

    st.session_state.chat_history.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            try:
                result = query_with_sources(question, top_k=top_k)

            # ── Bug fix 2: handle 429 rate limit gracefully ──
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate limit" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                    wait = _parse_retry_seconds(err_str)
                    st.warning(
                        f"⏳ Rate limit reached. Groq free tier allows 30 requests/min. "
                        f"Please wait **{wait} seconds** and try again."
                    )
                else:
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

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.session_state.eval_buffer.append(
            {
                "question": question,
                "answer": answer,
                "contexts": [s["text"] for s in sources],
            }
        )

if st.session_state.chat_history:
    if st.button("🗑️ Clear chat", use_container_width=False):
        st.session_state.chat_history = []
        st.rerun()