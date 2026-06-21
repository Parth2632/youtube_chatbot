# =========================
# IMPORTS
# =========================
import streamlit as st  # UI framework
import re  # For extracting video ID
import os  # For environment variables
from dotenv import load_dotenv  # Load .env file

# YouTube transcript API
from youtube_transcript_api import YouTubeTranscriptApi

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide"
)

# Load environment variables (.env file)
load_dotenv()


# =========================
# ICONS (inline SVG, lucide-style outline icons)
# =========================
def icon(name, size=18, color="currentColor"):
    """Return an inline SVG icon as an HTML string (lucide-style, stroke-based)."""
    icons = {
        "film": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="18" rx="2" ry="2"></rect><line x1="7" y1="3" x2="7" y2="21"></line><line x1="17" y1="3" x2="17" y2="21"></line><line x1="2" y1="9" x2="7" y2="9"></line><line x1="2" y1="15" x2="7" y2="15"></line><line x1="17" y1="9" x2="22" y2="9"></line><line x1="17" y1="15" x2="22" y2="15"></line></svg>''',
        "sun": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>''',
        "moon": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>''',
        "link": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path></svg>''',
        "play": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>''',
        "trash": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>''',
        "check-circle": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>''',
        "alert-circle": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>''',
        "file-text": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>''',
        "pointing": f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11V6a2 2 0 0 1 4 0v5"></path><path d="M13 10.5V4a2 2 0 0 1 4 0v8"></path><path d="M17 9.5a2 2 0 0 1 4 0V13a8 8 0 0 1-8 8h-2a8 8 0 0 1-7-4l-2.5-4.33a1.5 1.5 0 0 1 2.6-1.5L6 14"></path></svg>''',
    }
    return icons.get(name, "")


def icon_label(name, text, size=18, color="currentColor", gap="8px"):
    """Inline SVG icon + text label, vertically centered."""
    return f'''<span style="display:inline-flex;align-items:center;gap:{gap};">{icon(name, size, color)}<span>{text}</span></span>'''


# =========================
# CACHE MODELS (IMPORTANT FOR SPEED)
# =========================
@st.cache_resource
def get_embeddings():
    # Loads embedding model only once (faster)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_chat_model():
    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

    if not hf_token:
        st.error("Add HF_TOKEN in your .env file")
        st.stop()

    # Load LLM from HuggingFace
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",  # 72B — much better instruction following
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=llm)


# =========================
# EXTRACT VIDEO ID FROM URL
# =========================
def extract_video_id(url):
    # Regex to extract YouTube video ID
    pattern = r'(?:v=|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)

    if match:
        return match.group(1)  # Return video ID

    return None


# =========================
# FETCH TRANSCRIPT
# =========================
def fetch_transcript(video_id):
    # --- Stage 1: Try youtube-transcript-api (fast) ---
    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en', 'hi'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        if transcript.strip():
            return transcript, None
    except Exception as api_err:
        pass  # Fall through to yt-dlp fallback

    # --- Stage 2: Fallback to yt-dlp (bypasses IP blocks) ---
    try:
        import yt_dlp, requests

        ydl_opts = {
            'writeautomaticsub': True,
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
        }

        ydl = yt_dlp.YoutubeDL(ydl_opts)
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

        subtitles = info.get('subtitles') or {}
        auto_captions = info.get('automatic_captions') or {}

        # Pick best available language
        target_lang = None
        for lang in ['en', 'hi']:
            if lang in subtitles or lang in auto_captions:
                target_lang = lang
                break
        if not target_lang:
            all_langs = list(subtitles.keys()) + list(auto_captions.keys())
            target_lang = all_langs[0] if all_langs else None

        if not target_lang:
            return None, "No subtitles or captions found for this video."

        formats = subtitles.get(target_lang) or auto_captions.get(target_lang) or []

        # Prefer json3 format (clean structured data)
        json3_url = next((f['url'] for f in formats if f.get('ext') == 'json3'), None)
        if json3_url:
            data = requests.get(json3_url).json()
            transcript = "".join(
                seg.get('utf8', '')
                for event in data.get('events', [])
                for seg in event.get('segs', [])
                if seg.get('utf8')
            )
            if transcript.strip():
                return transcript.replace('\n', ' '), None

        # Fallback: parse VTT
        if formats:
            vtt_text = requests.get(formats[0]['url']).text
            lines = [
                re.sub(r'<[^>]*>', '', l.strip())
                for l in vtt_text.split('\n')
                if l.strip() and '-->' not in l
                and not l.startswith(('WEBVTT', 'Kind:', 'Language:'))
            ]
            # Remove consecutive duplicates
            deduped = [l for i, l in enumerate(lines) if i == 0 or l != lines[i - 1]]
            transcript = " ".join(deduped)
            if transcript.strip():
                return transcript, None

        return None, "Could not extract any transcript content from this video."

    except Exception as ytdl_err:
        return None, (
            f"Both transcript methods failed.\n\n"
            f"yt-dlp error: {ytdl_err}\n\n"
            "Tip: Try uploading a cookies.txt file (exported from your browser) to bypass YouTube bot detection."
        )



# =========================
# FORMAT DOCUMENTS (RAG)
# =========================
def format_docs(docs):
    # Combine retrieved chunks into one context string
    return "\n\n".join(doc.page_content for doc in docs)


# =========================
# SESSION STATE (APP MEMORY)
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "video_url" not in st.session_state:
    st.session_state.video_url = ""


# =========================
# THEME CSS INJECTION
# =========================
# NOTE: .streamlit/config.toml sets the app's base theme to a neutral grey DARK
# theme. That's what makes st.chat_input (and every other native widget) render
# correctly without fragile CSS overrides — config.toml is the only mechanism
# Streamlit fully respects for st.chat_input's internal styling.
theme_css = """
<style>
    [data-testid="stAlertContainer"] {
        margin-bottom: 14px !important;
    }
    .stButton {
        margin-top: 6px;
    }
    .stButton > button * {
        opacity: 1 !important;
    }
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)

# Icon color (single dark theme now, so this is fixed)
icon_color = "#e8e8e8"


# =========================
# SIDEBAR (INPUT SECTION)
# =========================
with st.sidebar:
    st.markdown(
        f'<h2>{icon_label("film", "YouTube RAG Chatbot", size=24, color=icon_color, gap="10px")}</h2>',
        unsafe_allow_html=True
    )
    st.divider()

    # --- URL Input ---
    st.markdown(
        f'<div style="margin-bottom:6px;font-weight:600;">{icon_label("link", "YouTube Video URL", size=16, color=icon_color, gap="6px")}</div>',
        unsafe_allow_html=True
    )
    video_url = st.text_input(
        "url",
        value=st.session_state.video_url,
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed"
    )

    # --- Action buttons side by side (single source of truth — no duplicate button) ---
    btn_col, clear_col = st.columns([3, 1])
    with btn_col:
        process_clicked = st.button(
            "Process Video", use_container_width=True, icon=":material/play_arrow:"
        )
    with clear_col:
        clear_clicked = st.button(
            "", help="Clear URL and reset", use_container_width=True, icon=":material/delete:"
        )

    if clear_clicked:
        st.session_state.video_url = ""
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.rerun()

    # Process the video when the single Process Video button is clicked
    if process_clicked:
        video_id = extract_video_id(video_url)

        if not video_id:
            st.error("Invalid YouTube URL")
        else:
            with st.spinner("Fetching transcript..."):

                transcript, error = fetch_transcript(video_id)

                if error:
                    st.error(error)
                else:
                    # Split transcript into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    docs = splitter.create_documents([transcript])

                    # Convert chunks into embeddings
                    embeddings = get_embeddings()

                    # Store in FAISS vector DB
                    vectorstore = FAISS.from_documents(docs, embeddings)

                    # Save in session
                    st.session_state.vectorstore = vectorstore
                    st.session_state.video_url = video_url
                    st.session_state.chat_history = []

                    st.success("Video processed successfully!")

                    # Optional: show preview
                    with st.expander("Transcript Preview", icon=":material/description:"):
                        st.write(transcript[:1500] + "...")


# =========================
# MAIN UI
# =========================
st.markdown(
    f'<h1>{icon_label("film", "YouTube RAG Chatbot", size=36, color=icon_color, gap="14px")}</h1>',
    unsafe_allow_html=True
)

# Show video if loaded
if st.session_state.video_url:
    st.video(st.session_state.video_url)


# =========================
# CHAT SECTION
# =========================
if st.session_state.vectorstore is not None:

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (modern UI)
    user_query = st.chat_input("Ask something about the video...")

    if user_query:
        # Save user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query
        })

        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                # Create retriever
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 6}
                )

                # Load LLM
                chat_model = get_chat_model()

                # Prompt for RAG
                prompt = PromptTemplate(
                    template="""
You are a YouTube assistant.

Answer ONLY using the transcript.
If not found, say "I couldn't find that in the video."

Context:
{context}

Question:
{question}

Answer:
""",
                    input_variables=["context", "question"]
                )

                # Build RAG pipeline
                chain = (
                    RunnableParallel({
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough()
                    })
                    | prompt
                    | chat_model
                    | StrOutputParser()
                )

                # Stream response (nice UX)
                full_response = ""

                for chunk in chain.stream(user_query):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # Save response
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:14px 18px;'
        f'border-radius:10px;background-color:#222222;'
        f'border:1px solid #333333;">'
        f'{icon("pointing", 20, icon_color)}'
        f'<span>Load a video first from the sidebar to start chatting.</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# =========================
# RESET BUTTON
# =========================
if st.button("Reset Chat", icon=":material/delete:"):
    st.session_state.chat_history = []
