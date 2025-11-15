# 🎯 Video Relevance Analyzer

A lightweight, YouTube transcript–based relevance engine that evaluates how closely a video matches its claimed title/topic.  
Built with **Streamlit, SentenceTransformers, BM25**, and a dual-mode transcript fetcher (**yt-dlp + Whisper fallback**).

---

## 🚀 Features

### **1. Dual Transcript Extraction**
- **Fast Mode**: Uses YouTube subtitles via yt-dlp  
- **Deep Mode**: Downloads audio + transcribes using **Whisper (base model)**

---

### **2. Hybrid RAG Relevance Model**
Combines multiple signals for robust relevance scoring:

- **Semantic embeddings** (dense similarity using MiniLM)
- **Sparse retrieval** scores (**BM25**)
- **Title expansion heuristic**
- **Overlapping semantic chunking**

➡️ Final output is a **weighted hybrid score (0–100%)**

---

### **3. Full Streamlit UI**
User can:

- Paste a YouTube URL + expected title  
- Choose transcript mode  
- Tune model settings (chunking, weighting)  
- View:
  - **Final relevance %**
  - **Top matching transcript segment**
  - Debug metrics & charts

---

## 📁 Project Structure

```
video-relevance-analyzer/
│
├── streamlit_app.py          # Main Streamlit UI
├── transcript_utils.py        # yt-dlp + Whisper transcript extractor
├── relevance_utils.py         # Embedding, chunking, title expansion
├── model.py                   # Hybrid relevance scoring engine
├── retrieval.py               # BM25 implementation
├── utils.py                   # Shared helpers
├── requirements.txt           # Dependencies
└── README.md                  # (this file)
```

---

## 🛠 Installation

```bash
git clone https://github.com/<your-user>/video-relevance-analyzer.git
cd video-relevance-analyzer
pip install -r requirements.txt
```

### Additional Requirements

#### **Node.js required** by yt-dlp for JSON3 subtitles

#### Optional: Whisper for deep transcript mode
```bash
pip install openai-whisper
```

---

## ▶️ Usage

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

### Steps
1. Enter YouTube URL  
2. Enter Expected Title / Topic  
3. Choose transcript mode:
   - Fast (YouTube captions)
   - Deep (Whisper)
4. Fetch transcript  
5. Run **Analyze Relevance**

### Output Includes:
- **Relevance % (0–100)**
- **Best matching chunk**
- **Dense & sparse scoring**
- Debug info (optional)

---

## 🧠 How the Relevance Model Works

### **1. Title Expansion**
- Extract keywords  
- Remove stopwords  
- Add paraphrase hints  
- Stabilizes scores for both BM25 & embeddings  

---

### **2. Semantic Chunking**
- Windowing by words  
- Default: **160 words, 30-word overlap**  
- Reduces noise from long transcripts  

---

### **3. Dense Similarity (Embeddings)**
Using:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

Computes vector similarity for each chunk.

---

### **4. Sparse Similarity (BM25)**
Custom BM25 implemented in `retrieval.py`.

---

### **5. Hybrid Score**

```
combined = 0.6 * dense + 0.4 * sparse
final_score = mean(top_k_combined_scores) * 100
```

---

## 📊 Example Output

```
Relevance Score: 82.7%

Top Matching Segment:
“… the speaker discusses how to build APIs using Postman …”
```

---

## 🧩 File-Level Summary

### `transcript_utils.py`
- yt-dlp fast transcript
- Whisper fallback
- Cleans captions
- Handles missing subtitles

### `relevance_utils.py`
- Title expansion
- Chunker
- Embedding helper
- Cosine similarity

### `model.py`
- Hybrid dense + sparse scoring
- Aggregates multi-signal scoring

### `retrieval.py`
- BM25 implementation

### `streamlit_app.py`
- Complete UI
- Settings panel
- Debug console

---

## 🧪 Testing

```python
from model import RelevanceModel

model = RelevanceModel()
result = model.compute_relevance("API Testing with Postman", transcript_text)
print(result)
```

---

## 📌 Roadmap
- Add local embedding support
- Integrate OpenAI or Gemini embeddings
- YouTube download caching
- Export PDF/JSON report
- Backend API endpoint

---

## 📄 License
MIT License - free to use and modify.

---

