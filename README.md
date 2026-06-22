# RAG Eval Dashboard

A production-ready **Retrieval-Augmented Generation (RAG)** app that lets you query your company documents using natural language — with a built-in evaluation dashboard to measure answer quality.

Built entirely on **free-tier APIs** with zero credit card required.

---

## Live Demo 
https://rag-eval-dashboard-1.streamlit.app/ 


> Upload your PDFs → Ask questions → Evaluate quality scores

![App Screenshot](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Free_Tier-F55036?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  💬 Q&A Chat  │  │ 📁 Doc Manager   │  │ 📊 Eval Dashboard│  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  │
└─────────┼────────────────────┼─────────────────────┼────────────┘
          │                    │                     │
          ▼                    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│   engine.py     │  │   engine.py     │  │   evaluator.py      │
│  Query Engine   │  │ Ingest Pipeline │  │  Scoring Engine     │
└────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘
         │                    │                       │
         ▼                    ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Core Components                           │
│                                                                  │
│  ┌─────────────────────┐    ┌──────────────────────────────┐   │
│  │   LlamaIndex Core   │    │        Groq API               │   │
│  │  VectorStoreIndex   │    │  llama-3.3-70b-versatile      │   │
│  │  SentenceSplitter   │    │  OpenAI-compatible endpoint   │   │
│  │  OpenAILike LLM     │    │  14,400 req/day (free)        │   │
│  └──────────┬──────────┘    └──────────────────────────────┘   │
│             │                                                    │
│  ┌──────────▼──────────┐    ┌──────────────────────────────┐   │
│  │    ChromaDB          │    │   HuggingFace Embeddings      │   │
│  │  Persistent local    │    │   BAAI/bge-small-en-v1.5     │   │
│  │  vector store        │    │   Runs 100% locally           │   │
│  └─────────────────────┘    └──────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Document Parsers                             │   │
│  │    pypdf (PDF)  │  docx2txt (DOCX)  │  built-in (TXT/MD) │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline — Step by Step

```
INGESTION
─────────
  PDF/DOCX/TXT/MD
       │
       ▼
  Text Extraction          pypdf (page-by-page) / docx2txt / read_text
       │
       ▼
  Chunking                 SentenceSplitter — 512 tokens, 64 overlap
       │
       ▼
  Embedding                BAAI/bge-small-en-v1.5 (local, no API)
       │
       ▼
  ChromaDB                 Persistent cosine-similarity vector store
       │
       ▼
  Dedup Check              SHA-256 hash per file → skip if already indexed


QUERY
─────
  User Question
       │
       ▼
  Embed Question           Same model as ingestion
       │
       ▼
  Vector Search            Top-K cosine similarity (default K=4)
       │
       ▼
  Context Assembly         Retrieved chunks → prompt context
       │
       ▼
  Groq LLM                 llama-3.3-70b-versatile (compact response mode)
       │
       ▼
  Answer + Sources         Answer text + chunk metadata (file, page, score)


EVALUATION
──────────
  Q&A pairs buffered from session
       │
       ▼
  Faithfulness Score       LLM checks: are claims supported by context?
       │
       ▼
  Context Utilization      LLM checks: was the context actually used?
       │
       ▼
  Radar Chart + Table      Plotly visualization + CSV export
```

---

## Tech Stack

| Layer | Tool | Cost |
|---|---|---|
| Frontend | Streamlit 1.35 | Free |
| LLM | Groq — `llama-3.3-70b-versatile` | Free (14,400 req/day) |
| Embeddings | `BAAI/bge-small-en-v1.5` via HuggingFace | Free (local) |
| RAG Framework | LlamaIndex Core 0.14 | Free |
| LLM Integration | `llama-index-llms-openai-like` → Groq endpoint | Free |
| Vector Store | ChromaDB 1.5 (persistent, local) | Free |
| PDF Parsing | pypdf 4.x | Free |
| DOCX Parsing | docx2txt | Free |
| Evaluation | Custom scoring via Groq API (no ragas dependency) | Free |

> **Why no ragas library?** Every version of ragas has a broken dependency chain on the current langchain/LangChain-Community ecosystem. The evaluation metrics are implemented directly using structured Groq API calls — same logic, zero dependency conflicts.

---

## Project Structure

```
kb_app/
│
├── app.py                      # Main page: Q&A chat interface
├── engine.py                   # RAG backend: ingestion + query
├── evaluator.py                # Evaluation: faithfulness + context utilization
├── config.py                   # Centralized config from .env
│
├── pages/
│   ├── 1_Document_Manager.py   # Upload and index documents
│   └── 2_RAGAS_Evaluation.py   # Evaluation dashboard
│
├── data/                       # Uploaded documents (gitignored)
├── chroma_db/                  # Vector store (gitignored)
│
├── .env.example                # Environment variable template
├── .env                        # Your actual keys (gitignored)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .streamlit/
    └── config.toml             # Dark theme + upload size settings
```

---

## Quick Start

### 1. Get a free Groq API key

Sign up at **https://console.groq.com/keys** — no credit card, instant access.

Free limits: **14,400 requests/day · 6,000 RPM · 128k context window**

### 2. Clone and configure

```bash
git clone https://github.com/surabhi-chandrakant/RAG-Eval-Dashboard.git
cd RAG-Eval-Dashboard

cp .env.example .env
# Open .env and set: GROQ_API_KEY=your_key_here
```

### 3. Install and run

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

App opens at **http://localhost:8501**

### 4. Docker (optional)

```bash
docker-compose up --build
```

---

## Usage

### Step 1 — Index your documents

Go to **📁 Document Manager** → upload PDF, DOCX, TXT, or MD files → click **Index Documents**.

- Ingestion is incremental — re-uploading the same file skips it (SHA-256 dedup)
- PDFs are parsed page-by-page for accurate source attribution
- Chunks are stored in ChromaDB on disk and persist across restarts

### Step 2 — Ask questions

Go to **💬 Q&A** → type any question → get an answer with:
- Source chunks showing which file and page the answer came from
- Similarity scores for each retrieved chunk
- Adjustable Top-K slider (1–10 chunks)

### Step 3 — Evaluate quality

Go to **📊 RAGAS Evaluation** → click **Run Evaluation** on buffered Q&A pairs.

**Metrics:**

| Metric | What it measures | Good score |
|---|---|---|
| Faithfulness | Are all claims in the answer grounded in retrieved context? Low = hallucination. | > 0.75 |
| Context Utilization | Was the retrieved context actually used to form the answer? Low = context ignored. | > 0.75 |

Results include a radar chart, per-question breakdown table, and CSV export.

---

## Configuration

All settings live in `.env`:

```env
# Required
GROQ_API_KEY=your_key_here

# Model (free options)
GROQ_MODEL=llama-3.3-70b-versatile   # smartest
# GROQ_MODEL=llama-3.1-8b-instant    # fastest
# GROQ_MODEL=mixtral-8x7b-32768      # longest context

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=4

# Storage
CHROMA_DB_PATH=./chroma_db
COLLECTION_NAME=company_kb
EMBED_MODEL=BAAI/bge-small-en-v1.5
```

---

## Key Design Decisions

**Groq instead of OpenAI/Gemini** — Groq's free tier is the most generous available: 14,400 req/day with no credit card. Gemini free tier has a token-per-minute cap that triggers quota errors on normal usage. OpenAI has no free tier.

**`llama-index-llms-openai-like` instead of `llama-index-llms-groq`** — The dedicated Groq package pins `llama-index-core<0.14` via its `openai-like` dependency, conflicting with everything else. Groq exposes an OpenAI-compatible REST API, so `OpenAILike` pointed at `https://api.groq.com/openai/v1` works identically with zero conflict.

**pypdf directly instead of `SimpleDirectoryReader`** — LlamaIndex's built-in PDF reader requires `llama-index-readers-file` which has its own dependency conflicts. Direct pypdf calls give page-by-page extraction with accurate page number metadata.

**Custom evaluation instead of ragas** — Every version of the ragas library has a broken import (`langchain_community.chat_models.vertexai`) on current langchain versions. The faithfulness and context utilization metrics are re-implemented as structured Groq API calls — identical scoring logic, zero library dependency.

**ChromaDB local over Pinecone** — No signup, no network latency, persists across restarts via `PersistentClient`. Scales comfortably to ~500k chunks on a single machine.

---

## Extending This Project

**Add more document types** — Edit `_load_file()` in `engine.py`. Add an `elif suffix == ".xlsx"` branch using `openpyxl`.

**Swap the LLM** — Change `GROQ_MODEL` in `.env` or point `OpenAILike` at any OpenAI-compatible endpoint (Ollama, Together AI, etc.) by updating `GROQ_API_BASE` in `engine.py`.

**Add authentication** — Wrap `app.py` with `streamlit-authenticator` or deploy behind Cloudflare Access.

**Cloud vector store** — Replace `ChromaVectorStore` in `engine.py` with `PineconeVectorStore` from `llama-index-vector-stores-pinecone`.

**Add more eval metrics** — Extend `evaluator.py` with additional `_ask_groq()` prompt functions for answer completeness, conciseness, or domain-specific criteria.

---

## Troubleshooting

**`GROQ_API_KEY is not set`** — Copy `.env.example` to `.env` and add your key.

**Dependency conflicts on install** — Always use a **fresh venv**. Accumulated packages from previous installs cause unresolvable conflicts.
```bash
rmdir /s /q venv          # Windows
rm -rf venv               # Mac/Linux
python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

**PDF returns raw binary instead of text** — Delete `chroma_db/` and re-index. Old chunks may contain binary garbage from a previous broken parser.

**Streamlit `Could not find page` error** — Page filenames must be ASCII only. Do not use emoji in filenames on Windows.

**`ModuleNotFoundError: No module named 'ragas'`** — Your `pages/2_RAGAS_Evaluation.py` is an old version. The current evaluator has no ragas dependency. Re-download the file from the repo.

---

## License

MIT — free to use, modify, and distribute.

---

## Author

**Surabhi Chandrakant Bhor**
[GitHub](https://github.com/surabhi-chandrakant) · [LinkedIn](https://linkedin.com/in/surabhi-chandrakant-bhor)

---

*Built with LlamaIndex · ChromaDB · Groq · Streamlit*
