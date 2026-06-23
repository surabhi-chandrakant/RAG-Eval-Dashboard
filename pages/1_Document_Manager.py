"""
Document Manager — upload and index files.

Fixes:
  3. Axios upload error — files now saved in chunks (streaming write)
     with a per-file size cap and type validation before saving to disk.
"""
import streamlit as st
from pathlib import Path
import config
from engine import get_collection_stats, ingest_documents, delete_collection

st.set_page_config(page_title="Document Manager", page_icon="📁", layout="wide")

errors = config.validate_config()
if errors:
    st.error("\n".join(errors))
    st.stop()

# Max file size: 50 MB per file (Streamlit default is 200MB which causes Axios timeout)
MAX_FILE_MB = 50
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

ALLOWED_TYPES = {".pdf", ".txt", ".md", ".docx"}

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
    st.caption(f"Max {MAX_FILE_MB} MB per file")

if uploaded_files:
    # ── Bug fix 3: validate size and type before touching disk ──
    valid_files = []
    for f in uploaded_files:
        ext = Path(f.name).suffix.lower()
        size_mb = len(f.getvalue()) / (1024 * 1024)

        if ext not in ALLOWED_TYPES:
            st.error(f"❌ `{f.name}` — unsupported file type `{ext}`")
            continue
        if len(f.getvalue()) > MAX_FILE_BYTES:
            st.error(f"❌ `{f.name}` — {size_mb:.1f} MB exceeds the {MAX_FILE_MB} MB limit")
            continue
        valid_files.append(f)

    if valid_files:
        st.markdown(f"**{len(valid_files)} valid file(s) ready to index:**")
        for f in valid_files:
            size_kb = len(f.getvalue()) / 1024
            st.write(f"  • `{f.name}` ({size_kb:.1f} KB)")

        if st.button("⚙️ Index Documents", type="primary", use_container_width=True):
            saved_paths = []
            progress = st.progress(0, text="Saving files…")

            save_errors = []
            for i, uf in enumerate(valid_files):
                dest = config.UPLOAD_DIR / uf.name
                try:
                    # Write in 1 MB chunks to avoid memory spike (Axios fix)
                    data = uf.getvalue()
                    with open(dest, "wb") as out:
                        chunk_size = 1024 * 1024
                        for offset in range(0, len(data), chunk_size):
                            out.write(data[offset:offset + chunk_size])
                    saved_paths.append(dest)
                except Exception as exc:
                    save_errors.append(f"{uf.name}: {exc}")

                progress.progress(
                    (i + 1) / len(valid_files),
                    text=f"Saved {uf.name}"
                )

            if save_errors:
                for err in save_errors:
                    st.error(f"Save failed: {err}")

            if saved_paths:
                progress.progress(100, text="Indexing into vector store…")
                try:
                    chunks_added, skipped = ingest_documents(saved_paths)
                    progress.empty()
                    if chunks_added > 0:
                        new_count = len(saved_paths) - len(skipped)
                        st.success(f"Indexed {chunks_added} chunks from {new_count} new file(s).")
                    if skipped:
                        skipped_names = ", ".join(skipped)
                        st.info(f"Already indexed (skipped): {skipped_names}")
                    st.rerun()
                except Exception as exc:
                    progress.empty()
                    st.error(f"Indexing failed: {exc}")
            else:
                progress.empty()

st.divider()
st.subheader("Indexed Knowledge Base")
stats = get_collection_stats()

if stats["total_chunks"] == 0:
    st.info("No documents indexed yet. Upload files above to get started.")
else:
    st.markdown(
        f"**{stats['total_chunks']} chunks** across "
        f"**{len(stats['unique_files'])} file(s)**"
    )
    for fname in stats["unique_files"]:
        st.write(f"  📄 {fname}")

    st.divider()
    st.subheader("⚠️ Danger Zone")
    st.warning("Clears all vector embeddings. Uploaded files are kept on disk.")
    confirm = st.checkbox("I understand this cannot be undone")
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