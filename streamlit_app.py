# streamlit_app.py
import streamlit as st
from transcript_utils import get_transcript
from model import RelevanceModel
import json

st.set_page_config(page_title="Video Relevance Analyzer (Hybrid RAG)", layout="wide")
st.title("🎯 Video Relevance Analyzer — Hybrid RAG")

col1, col2 = st.columns([3,1])

with col1:
    url = st.text_input("YouTube Video URL")
    title = st.text_input("Video Title / Topic (expected/claimed)")

with col2:
    st.write("Transcript Mode")
    mode = st.radio("", ["Fast (YouTube Subtitles)", "Deep (Whisper)"])
    mode_key = "fast" if mode.startswith("Fast") else "deep"
    st.write("---")
    st.write("Model settings")
    # small interactive knobs
    chunk_size = st.number_input("Chunk size (words)", value=160, step=10)
    overlap = st.number_input("Chunk overlap (words)", value=30, step=5)
    dense_w = st.slider("Dense weight", 0.0, 1.0, 0.6)
    sparse_w = 1.0 - dense_w
    topk = st.slider("Aggregate top-k", 1, 5, 3)

if st.button("Fetch Transcript"):
    with st.spinner("Fetching transcript..."):
        txt, err = get_transcript(url, mode=mode_key)
    if err:
        st.error(err)
    else:
        st.session_state["transcript"] = txt
        st.success("Transcript fetched and stored.")

if "transcript" in st.session_state:
    st.subheader("📝 Transcript (editable)")
    transcript = st.text_area("", st.session_state["transcript"], height=300)
else:
    transcript = ""

if st.button("Analyze Relevance") and transcript and title:
    with st.spinner("Computing relevance..."):
        model = RelevanceModel(
            chunk_size_words=int(chunk_size),
            overlap=int(overlap),
            dense_weight=float(dense_w),
            sparse_weight=float(sparse_w),
            top_k=int(topk),
        )
        result = model.compute_relevance(title, transcript)

    st.subheader("📊 Relevance Score")
    st.metric("Relevance (%)", result.get("score", 0.0))
    st.write("**Top matching segment (preview):**")
    st.code(result.get("top_chunk", "")[:800])

    st.write("---")
    st.subheader("🧾 Debug details (for tuning)")
    details = result.get("details", {})
    st.write(f"Number of chunks: {details.get('num_chunks', 'N/A')}")
    cols = st.columns(3)
    with cols[0]:
        st.write("Dense (embed) scores (first 10)")
        st.write(details.get("dense_scores", [])[:10])
    with cols[1]:
        st.write("Sparse (BM25) scores (first 10)")
        st.write(details.get("sparse_scores", [])[:10])
    with cols[2]:
        st.write("Combined scores (first 10)")
        st.write(details.get("combined_scores", [])[:10])

    st.write("---")
    if st.checkbox("Show full debug JSON"):
        st.json(result)
else:
    if st.button("Analyze Relevance (disabled)"):
        st.warning("Enter both Title and Transcript first.")
