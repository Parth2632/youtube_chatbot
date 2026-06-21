from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not hf_token:
    raise ValueError("Set HF_TOKEN or HUGGINGFACEHUB_ACCESS_TOKEN in your .env file.")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

chat_model = ChatHuggingFace(llm=llm)

# ---- YouTube + RAG pipeline ----
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ✅ Valid video ID (has transcript)
video_id = "dQw4w9WgXcQ"   # famous video, transcripts available

transcript = ""

try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    transcript = " ".join(chunk.text for chunk in transcript_list)
except Exception as e:
    print(f"Error fetching transcript: {e}")

if not transcript:
    print("No transcript could be retrieved. Exiting script.")
    exit(1)

# ---- Chunking ----
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# ---- Embeddings ----
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---- Vector DB ----
vectorstore = FAISS.from_documents(chunks, embeddings)

# ---- Retriever ----
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# ---- Prompt ----
prompt = PromptTemplate(
    template="""You are a helpful youtube chatbot. 
Answer ONLY from the provided transcript context. 
If the context is insufficient, just say you don't know.

context: {context}
question: {question}""",
    input_variables=["context", "question"]
)

# ---- Runnable Pipeline ----
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parallel_chain = RunnableParallel({
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

# Final RAG chain
rag_chain = parallel_chain | prompt | chat_model | parser

# ---- Ask Question ----
response = rag_chain.invoke("What is this video about?")
print(response)