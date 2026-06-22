"""Document Manager — upload and index files."""
import streamlit as st
from pathlib import Path
import config
from engine import get_collection_stats, ingest_documents, delete_collection

st.set_page_config(page_title="Document Manager", page_icon="📁", layout="wide")

errors = config.validate_config()
if errors:
    st.error("\n".join(errors))
    st.stop()

with st.sidebar:
    st.title("📁 Document Manager")
    stats = get_collection_stats()
    st.metric("Indexed Chunks", stats["total_chunks"])
    st.metric("Unique Files", len(stats["unique_files"]))

st.title("📁 Document Manager")
st.caption("Upload documents to index them into the knowledge base.")

st.subheader("Upload Documents")
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with col2:
    st.markdown("**Supported formats**")
    st.markdown("PDF, TXT, MD, DOCX")

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) selected**")
    for f in uploaded_files:
        size_kb = len(f.getvalue()) / 1024
        st.write(f"  • `{f.name}` ({size_kb:.1f} KB)")

    if st.button("⚙️ Index Documents", type="primary", use_container_width=True):
        saved_paths = []
        progress = st.progress(0, text="Saving files…")
        for i, uf in enumerate(uploaded_files):
            dest = config.UPLOAD_DIR / uf.name
            dest.write_bytes(uf.getvalue())
            saved_paths.append(dest)
            progress.progress((i + 1) / len(uploaded_files), text=f"Saved {uf.name}")
        progress.progress(100, text="Indexing…")
        try:
            chunks_added, skipped = ingest_documents(saved_paths)
            progress.empty()
            if chunks_added > 0:
                st.success(f"Indexed {chunks_added} chunks from {len(saved_paths) - len(skipped)} new file(s).")
            if skipped:
                skipped_names = ", ".join(skipped)
                st.info(f"Skipped already-indexed: {skipped_names}")
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"Indexing failed: {exc}")

st.divider()
st.subheader("Indexed Knowledge Base")
stats = get_collection_stats()
if stats["total_chunks"] == 0:
    st.info("No documents indexed yet.")
else:
    st.markdown(f"**{stats['total_chunks']} chunks** across **{len(stats['unique_files'])} file(s)**")
    for fname in stats["unique_files"]:
        st.write(f"  📄 {fname}")

    st.divider()
    st.subheader("⚠️ Danger Zone")
    confirm = st.checkbox("I understand this clears all vector embeddings")
    if st.button("🗑️ Clear Entire Index", type="secondary", disabled=not confirm):
        delete_collection()
        st.success("Index cleared.")
        st.rerun()

st.divider()
with st.expander("⚙️ Current Settings"):
    st.code(
        f"Model         : {config.GROQ_MODEL}\n"
        f"Embed model   : {config.EMBED_MODEL}\n"
        f"Chunk size    : {config.CHUNK_SIZE} tokens\n"
        f"Chunk overlap : {config.CHUNK_OVERLAP} tokens\n"
        f"Vector store  : ChromaDB (local)\n"
        f"Collection    : {config.COLLECTION_NAME}"
    )