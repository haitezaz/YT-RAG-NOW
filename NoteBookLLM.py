import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import os

load_dotenv()

# Load models and tools
llm = ChatOpenAI(model="gpt-3.5-turbo")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
parser = StrOutputParser()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer only from context below. If context is insufficient, just say you don't know.

Query: {query}
Context: {context}"""
)

import os
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import yt_dlp
import webvtt

# Your existing extract_video_id function
def extract_video_id(youtube_url: str) -> str:
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/embed/")[1].split("/")[0]
        if parsed_url.path.startswith("/v/"):
            return parsed_url.path.split("/v/")[1].split("/")[0]
    if parsed_url.hostname in ["youtu.be"]:
        return parsed_url.path.lstrip("/").split("/")[0]
    return None

# Your yt_dlp + webvtt method to download and parse subtitles
def download_and_parse_subtitles(video_url, output_dir="subtitles"):
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        video_id = info.get("id")

    vtt_file = Path(f"{output_dir}/{video_id}.en.vtt")
    if not vtt_file.exists():
        raise FileNotFoundError("Subtitle file not found after download attempt.")

    full_text = ""
    for caption in webvtt.read(str(vtt_file)):
        full_text += caption.text + " "

    return full_text.strip()

# Your modified main function
def getTranscriptAndMakeVectorStore(youtubeVideoLink):
    video_id = extract_video_id(youtubeVideoLink)
    if not video_id:
        raise ValueError("Invalid YouTube URL or could not extract video ID.")

    try:
        # Download and parse subtitles (auto or manual)
        full_transcript = download_and_parse_subtitles(youtubeVideoLink)
    except FileNotFoundError:
        raise ValueError("Transcript not available or failed to fetch via yt_dlp subtitles.")
    except Exception as e:
        raise ValueError(f"Unexpected error fetching subtitles: {e}")

    # Assuming you have a text splitter and embeddings model set up
    chunks = text_splitter.split_text(full_transcript)
    documents = [Document(page_content=chunk) for chunk in chunks]
    db = FAISS.from_documents(documents, embeddings_model)
    return db


# -------------------- Streamlit UI ----------------------

st.set_page_config(page_title="YouTube Q&A", layout="centered")
st.title("🎥 YouTube Video QA with LangChain")

youtube_url = st.text_input("Enter a YouTube video URL:")

if youtube_url:
    db = getTranscriptAndMakeVectorStore(youtube_url)
    if db:
        retriever = db.as_retriever()
        
        query = st.text_input("Enter your question:")
        if query:
            chain1 = RunnablePassthrough()
            chain2 = retriever | parser
            parallelChain = RunnableParallel({
                "query": chain1,
                "context": chain2
            })
            finalChain = parallelChain | prompt | llm | parser

            if st.button("Ask"):
                with st.spinner("Thinking..."):
                    answer = finalChain.invoke(query)
                    st.success(answer)