🎯 Video Relevance Analyzer — Hybrid RAG Engine

A lightweight YouTube transcript–based relevance engine that evaluates how closely a video matches its claimed title/topic.

Built with Streamlit, SentenceTransformers, BM25, and a dual-mode transcript fetcher (yt-dlp + Whisper fallback).

🚀 Features
1️⃣ Dual Transcript Extraction

✔ Fast Mode (default) – YouTube subtitles via yt-dlp
✔ Deep Mode – Automatic Whisper (base) transcription when captions aren't available

2️⃣ Hybrid RAG Relevance Model

Combines three complementary signals:

Component	Description
Semantic Embeddings	Dense similarity (MiniLM)
Sparse Retrieval (BM25)	Lexical relevance, keyword alignment
Title Expansion	Keyword-based augmentation improves retrieval

Additional improvements:

Overlapping semantic chunking

Normalized hybrid scoring

Weighted combination → final relevance % (0–100)

3️⃣ Full Streamlit User Interface

Enter YouTube URL & expected title/topic

Fetch transcript automatically

Optional: Deep transcript via Whisper

Tune model settings (chunk size, weights)

View:

✔ Final relevance percentage
✔ Best-matching transcript segment
✔ Debug dashboard (dense, sparse, hybrid)

📁 Project Structure
📦 video-relevance-analyzer
│
├── streamlit_app.py        # Streamlit UI
├── transcript_utils.py     # yt-dlp + Whisper transcript extractor
├── relevance_utils.py      # Chunking, embeddings, title expansion
├── model.py                # Hybrid scoring engine
├── retrieval.py            # BM25 implementation
├── utils.py                # Shared utility functions
├── requirements.txt        # Dependencies
└── README.md               # (this file)

🛠 Installation
git clone https://github.com/<your-username>/video-relevance-analyzer.git
cd video-relevance-analyzer
pip install -r requirements.txt

Additional Requirements
✔ Node.js

Required by yt-dlp for parsing JSON3 subtitles
https://nodejs.org/

✔ Whisper (optional: only for Deep Mode)
pip install openai-whisper

▶️ Usage

Start Streamlit:

streamlit run streamlit_app.py

Steps to Analyze:

Paste YouTube URL

Enter expected/claimed title

Choose Fast or Deep transcript mode

Fetch transcript

Analyze relevance

You’ll receive:

Final relevance %

Top matching transcript chunk

Internal debug metrics (optional)

🧠 How the Relevance Model Works
1️⃣ Title Expansion

Advanced text augmentation:

Extract keywords

Remove stopwords

Add paraphrased cues

This stabilizes both BM25 & embedding relevance.

2️⃣ Semantic Chunking

Window-based splitting

Default = 160 words

Overlap = 30 words

Prevents noise from very long transcripts.

3️⃣ Dense Similarity (Embeddings)

Using:

sentence-transformers/all-MiniLM-L6-v2


Computes cosine similarity between:

expanded title ↔ each transcript chunk

4️⃣ Sparse Similarity (BM25)

Lexical match scoring.

Custom BM25 engine:

TF normalization

IDF weighting

Longer transcript handling

5️⃣ Hybrid Score
combined = 0.6 * dense + 0.4 * sparse
final_score = mean(top_k_scores) * 100


Produces an interpretable 0–100 relevance score.

📊 Example Output

Relevance Score: 82.7%
Top Segment:

“... the speaker discusses how to build APIs using Postman and compares it with...”

🧩 File-Level Summary
transcript_utils.py

yt-dlp transcript extractor (json3, vtt, srt)

Whisper fallback mode

Cleans + normalizes subtitles

Fully failure-safe

relevance_utils.py

Title expansion logic

Semantic chunking

Embedding helpers + caching

Cosine similarity utilities

model.py

Hybrid relevance computation

Dense + sparse normalization

Top-K aggregation strategy

retrieval.py

Pure Python BM25

Efficient term + frequency handling

Stable scoring across varied chunk lengths

streamlit_app.py

Clean, intuitive UI

Transcript viewer

Relevance analyzer

Debug metrics for power users

🧪 Manual Testing
from model import RelevanceModel

model = RelevanceModel()
result = model.compute_relevance("API Testing with Postman", transcript_text)
print(result)

📌 Roadmap

Local embedding model support

Optional OpenAI/Gemini embeddings

Download throttling + caching

Export full PDF/JSON reports

REST API backend

Full Chrome Extension

📄 License

MIT License — free for personal and commercial use.
