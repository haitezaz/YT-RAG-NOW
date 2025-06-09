🎥 YouTube Real-Time RAG
Interact with YouTube videos using GPT-3.5 in real time!
This app uses Retrieval-Augmented Generation (RAG) to answer your questions based on the transcript of any public YouTube video.

🔍 Features
🔗 Input any YouTube link

📝 Extract subtitles using yt_dlp (manual or auto-generated)

📚 Split transcript into semantic chunks

🧠 Generate OpenAI Embeddings

⚡ Store vectors in FAISS for fast similarity search

🤖 Use GPT-3.5 (via LangChain) to answer questions based on transcript context

🖼️ Simple Streamlit UI

🚀 Demo
Coming soon – or run locally with the instructions below.

🧰 Tech Stack
Tool	Purpose
Python	Main programming language
Streamlit	Web interface
yt_dlp	Download YouTube subtitles
webvtt	Parse .vtt subtitle files
FAISS	Store and retrieve dense vector chunks
OpenAI API	GPT-3.5 for Q&A and embeddings
LangChain	Chaining together the RAG components

🛠️ Setup Instructions
1. Clone the Repo
bash
Copy
Edit
git clone https://github.com/yourusername/youtube-real-time-rag.git
cd youtube-real-time-rag
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Add Environment Variables
Create a .env file with your OpenAI API key:

env
Copy
Edit
OPENAI_API_KEY=your_openai_key_here
4. Run the App
bash
Copy
Edit
streamlit run app.py
