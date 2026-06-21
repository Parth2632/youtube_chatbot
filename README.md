# 🎬 YouTube RAG Chatbot

Ask questions about any YouTube video and get answers grounded in what was actually said — not a hallucinated summary.

This app pulls a video's transcript, chunks and embeds it, and answers your questions using **Retrieval-Augmented Generation (RAG)** instead of relying on an LLM's general knowledge. If the answer isn't in the video, it tells you that — it doesn't make one up.

---

## Why this exists

Most "chat with a video" demos just paste the whole transcript into a prompt and hope for the best. That breaks down fast: long videos blow past context limits, the model loses track of where information came from, and answers drift into plausible-sounding nonsense once the transcript gets long enough.

This project does it the way a production RAG system actually should:

- **Chunk and embed**, don't dump — the transcript is split into overlapping ~1000-character chunks and embedded with `sentence-transformers/all-MiniLM-L6-v2`, so retrieval stays precise even on multi-hour videos.
- **Retrieve, then generate** — a FAISS vector store finds the 6 most relevant chunks for each question, and only those are passed to the LLM. The model never sees content it doesn't need.
- **Constrained generation** — the prompt explicitly instructs the model to answer only from retrieved context and say so when it can't find something, instead of guessing.
- **Resilient transcript fetching** — a two-stage pipeline tries the YouTube Transcript API first (fast), then automatically falls back to `yt-dlp` caption extraction if the video is geo-blocked or rate-limited, parsing both `json3` and raw `VTT` formats.

## How it works

```
YouTube URL
     │
     ▼
Extract video ID  →  Fetch transcript (API → yt-dlp fallback)
     │
     ▼
Split into chunks (1000 chars, 200 overlap)
     │
     ▼
Embed chunks  →  Store in FAISS vector index
     │
     ▼
User question  →  Retrieve top-6 relevant chunks  →  Qwen2.5-72B-Instruct  →  Grounded answer
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) | Fast iteration, native chat components |
| Orchestration | [LangChain](https://www.langchain.com) (LCEL) | Composable retrieval → prompt → generation pipeline |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight, fast, strong semantic retrieval for its size |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) | In-memory, no external DB needed for a single-session app |
| LLM | Qwen2.5-72B-Instruct via Hugging Face Inference Endpoints | Strong instruction-following at a fraction of frontier-model cost |
| Transcript fetching | `youtube-transcript-api` + `yt-dlp` fallback | Survives the API's frequent IP/rate-limit blocks |

## Features

- 🔍 **Source-grounded Q&A** — every answer traces back to the actual transcript
- ⚡ **Streamed responses** — tokens appear as they're generated, not after a long wait
- 🔄 **Automatic fallback transcript fetching** — keeps working when YouTube blocks the primary API
- 🎨 **Custom dark UI** — neutral-grey theme with hand-built SVG iconography (no emoji, no default Streamlit chrome)
- 💬 **Persistent chat history** within a session, with one-click reset
- 🗂️ **Per-video vector indexing** — switching videos rebuilds the index cleanly

## Getting started

### Prerequisites
- Python 3.10–3.11
- A free [Hugging Face](https://huggingface.co/settings/tokens) access token

### Setup

```bash
git clone <your-repo-url>
cd youtube-rag-chatbot
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

### Run

```bash
streamlit run app.py
```

Paste a YouTube URL into the sidebar, hit **Process Video**, and start asking questions.

## Project structure

```
.
├── app.py                  # Application entry point (UI + RAG pipeline)
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Native dark theme configuration
└── .env                    # HF_TOKEN (not committed)
```

## Design decisions worth noting

- **Theming via `config.toml`, not CSS overrides** — Streamlit's native theme system is the only mechanism that reliably styles every internal component (including `st.chat_input`); CSS injection alone can't fully reach it.
- **Caching with `@st.cache_resource`** — the embedding model and LLM client are loaded once per server process, not on every rerun, keeping the app responsive.
- **Graceful degradation** — transcript fetching never just fails silently; it explains what went wrong and what to try next (e.g. exporting a `cookies.txt` for bot-detection bypass).

## Possible extensions

- Swap FAISS for a persistent vector store (e.g. Chroma, Pinecone) to support multi-video libraries across sessions
- Add timestamp-linked citations so answers jump to the relevant moment in the video
- Support playlist-level ingestion for cross-video Q&A
- Add an evaluation harness (e.g. RAGAS) to measure retrieval precision/recall
