# streamlit_app.py
import streamlit as st
from transcript_utils import get_transcript
from model import RelevanceModel
import json

st.set_page_config(page_title="Video Relevance Analyzer — Hybrid RRF", layout="wide")
st.title("🎯 Video Relevance Analyzer — Hybrid RRF Model")

# ---------------------------------------------------------
# UI Inputs
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    url = st.text_input("YouTube Video URL", "")
    title = st.text_input("Expected Video Title / Topic", "")

with col2:
    st.write("Transcript Mode")
    mode = st.radio("", ["Fast (YouTube Subtitles)", "Deep (Whisper)"])
    mode_key = "fast" if mode.startswith("Fast") else "deep"

    st.write("---")
    st.write("⚙️ Model Settings")

    model_name = st.selectbox(
        "Embedding Model",
        [
            "intfloat/e5-large-v2",
            "BAAI/bge-large-en",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        index=0
    )

    chunk_size = st.number_input("Chunk size (words)", value=80, step=10)
    overlap = st.number_input("Chunk overlap (words)", value=20, step=5)
    topk = st.slider("Aggregate Top-k Chunks", 1, 6, 3)
    rrf_k = st.slider("RRF k-value (fusion softness)", 10, 200, 60)

# ---------------------------------------------------------
# TRANSCRIPT FETCHING
# ---------------------------------------------------------
if st.button("Fetch Transcript"):
    with st.spinner("Fetching transcript..."):
        txt, err = get_transcript(url, mode=mode_key)

    if err:
        st.error(err)
    else:
        st.session_state["transcript"] = txt
        st.success("Transcript fetched successfully.")

# Show transcript box if present
if "transcript" in st.session_state:
    st.subheader("📝 Transcript (editable)")
    transcript = st.text_area("Transcript", st.session_state["transcript"], height=300)
else:
    transcript = ""

# ---------------------------------------------------------
# RUN RELEVANCE ANALYSIS
# ---------------------------------------------------------
if st.button("Analyze Relevance") and transcript.strip() and title.strip():
    model = RelevanceModel(
        embed_model=model_name,
        chunk_size_words=int(chunk_size),
        overlap=int(overlap),
        top_k=int(topk),
        rrf_k=float(rrf_k),
        use_cache=True,
    )

    with st.spinner("Computing semantic relevance..."):
        result = model.compute_relevance(title, transcript)

    # ---------------------------------------------------------
    # RESULTS SECTION
    # ---------------------------------------------------------
    st.subheader("📊 Relevance Score")
    st.metric("Relevance (%)", result.get("score", 0.0))

    st.write("### 🔍 Top Matching Transcript Segment")
    st.code(result.get("top_chunk", ""), language="text")

    st.write("---")
    st.subheader("🧾 Debug Details")

    details = result.get("details", {})
    st.write(f"**Number of chunks:** {details.get('num_chunks', 'N/A')}")
    st.write(f"**Expanded Title:** `{details.get('expanded_title', '')}`")
    st.write(f"**Expansion Keywords:** {details.get('expansion_keywords', [])}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Dense Scores** (first 10)")
        st.write(details.get("dense_scores", [])[:10])

    with c2:
        st.write("**Sparse Scores** (first 10)")
        st.write(details.get("sparse_scores", [])[:10])

    with c3:
        st.write("**RRF Scores** (first 10)")
        st.write(details.get("rrf_scores", [])[:10])

    st.write("---")
    if st.checkbox("Show Full Debug JSON"):
        st.json(result)

else:
    if st.button("Analyze Relevance (disabled)"):
        st.warning("Please provide both Title and Transcript before running the analysis.")
