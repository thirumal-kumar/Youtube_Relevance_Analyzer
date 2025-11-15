**🎯 Video Relevance Analyzer — Hybrid RAG
**
A lightweight, YouTube transcript–based relevance engine that evaluates how closely a video matches its claimed title/topic.
Built with Streamlit, SentenceTransformers, BM25, and a dual-mode transcript fetcher (yt-dlp + Whisper fallback).

**🚀 Features**
**1. Dual Transcript Extraction
**
**Fast Mode:** Uses YouTube subtitles via yt-dlp

**Deep Mode:** Downloads audio + transcribes using Whisper (base model)

**2. Hybrid RAG Relevance Model**

**Combines:**

Semantic embeddings (dense similarity using MiniLM)

Sparse retrieval scores (BM25)

Title expansion heuristic

Overlapping semantic chunking

Weighted hybrid score → final % relevance (0–100)

**3. Full Streamlit UI**

Paste YouTube URL + expected title

Tune model settings: chunk size, overlap, dense/sparse weights

**Get:

Final relevance %**

Top matching transcript segment

Debug dashboards for dense/sparse/hybrid scores

**📁 Project Structure**
📦 video-relevance-analyzer
│
├── streamlit_app.py          # Main Streamlit UI
├── transcript_utils.py       # yt-dlp + Whisper transcript extractor
├── relevance_utils.py        # Embedding, chunking, title expansion
├── model.py                  # Hybrid relevance scoring engine
├── retrieval.py              # BM25 implementation
├── utils.py                  # Small shared utility functions
├── requirements.txt          # Dependencies
└── README.md                 # (this file)

**🛠 Installation**
git clone https://github.com/<your-username>/video-relevance-analyzer.git
cd video-relevance-analyzer
pip install -r requirements.txt

**Additional Requirements**

Node.js required by yt-dlp for JSON3 subtitles

Optional: Whisper for deep transcript mode

pip install openai-whisper

**▶️ Usage**
Start the Streamlit app
streamlit run streamlit_app.py

**Steps**

Enter YouTube URL

Enter Expected Title / Topic

Choose transcript mode:

Fast (YouTube captions)

Deep (Whisper)

Fetch transcript → Edit if needed

Run Analyze Relevance

You’ll get:

Final relevance percentage

Highest-matching transcript chunk

Debug metrics (dense, sparse, combined scores)

**🧠 How the Relevance Model Works**
**1. Title Expansion
**
Deterministic text augmentation:

Extract keywords

Remove stopwords

Add paraphrase signals

Stabilizes BM25 + embedding relevance

**2. Semantic Chunking**

Word-level windowing

Default: 160 words, 30-word overlap

Prevents noisy scoring of huge transcripts

**3. Dense Similarity (Embeddings)**

Using SentenceTransformer("all-MiniLM-L6-v2"):

Compute vector similarity between title expansion & each chunk
**
4. Sparse Similarity (BM25)**

Custom BM25 over chunks:

Measures lexical match strength

Complements semantic embeddings

**5. Hybrid Score**
combined = 0.6 * dense + 0.4 * sparse
final_score = mean(top_k_combined_scores) * 100

**6. Result**

Easy-to-interpret score (0–100)

Top matching transcript passage

Optional detailed debugging

**📊 Example Output
Relevance Score: 82.7%**

**Top Matching Segment:**
"… the speaker discusses how to build APIs using Postman and compares it with..."

**🧩 File-Level Summary**
transcript_utils.py

Fast transcript extractor via yt-dlp (supports json3, vtt, srt)

Whisper fallback mode for deep transcription

Robust cleaning & error conditions

relevance_utils.py

Title expansion logic

Semantic chunker

Embedding helper with disk caching

Cosine similarity + normalization utilities

model.py

The full hybrid model combining all components

Dense + sparse normalized scoring

Aggregation via top-k strategy

retrieval.py

Custom BM25 implementation

Efficient term frequency, IDF, normalization

streamlit_app.py

Clean UI

Interactive knobs for tuning

Debug panel for power users

🧪 Testing

To manually test the model:

from model import RelevanceModel

model = RelevanceModel()
result = model.compute_relevance("API Testing with Postman", transcript_text)
print(result)

📌 Roadmap (optional)

 Add local embedding model support

 Integrate OpenAI or Gemini embeddings

 Add YouTube download throttling & caching

 Export full report (PDF/JSON)

 Build backend API for programmatic usage

**📄 License**

MIT License — use freely.
