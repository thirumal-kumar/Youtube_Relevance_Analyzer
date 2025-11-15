# model.py
import numpy as np
from typing import Dict, Any
from relevance_utils import (
    chunk_text_semantic,
    expand_title,
    embed_texts,
    cosine_sim_matrix,
    normalize_array,
)
from retrieval import BM25  # uses your existing retrieval.py BM25
import math

class RelevanceModel:
    """
    Hybrid RAG Relevance Model (Dense + Sparse).
    - Title expansion via expand_title()
    - Semantic chunking of transcript
    - Dense embeddings similarity (sentence-transformers)
    - BM25 sparse scores (retrieval.BM25)
    - Hybrid combine: 0.6 * dense + 0.4 * sparse (both normalized)
    - Aggregate: mean(top_k) where top_k = min(3, num_chunks)
    """

    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        chunk_size_words: int = 160,
        overlap: int = 30,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        top_k: int = 3,
    ):
        self.embed_model = embed_model
        self.chunk_size_words = chunk_size_words
        self.overlap = overlap
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.top_k = top_k

    def compute_relevance(self, title: str, transcript: str) -> Dict[str, Any]:
        """
        Main entry. Returns:
          {
            "score": float (0-100),
            "top_chunk": str,
            "details": {
               "num_chunks": int,
               "dense_scores": [...],
               "sparse_scores": [...],
               "combined_scores": [...],
            }
          }
        """
        # sanitize inputs
        if not title:
            return {"score": 0.0, "reason": "empty title", "top_chunk": ""}
        if not transcript or not transcript.strip():
            return {"score": 0.0, "reason": "empty transcript", "top_chunk": ""}

        # 1) chunk transcript
        chunks = chunk_text_semantic(transcript, chunk_size_words=self.chunk_size_words, overlap=self.overlap)
        if not chunks:
            return {"score": 0.0, "reason": "no chunks produced", "top_chunk": ""}

        # 2) expand title to richer topic string for embedding and BM25
        expanded = expand_title(title)

        # 3) compute embeddings: title_topic + chunks
        # We'll compute title embedding and chunk embeddings
        texts_for_embed = [expanded] + chunks
        embeddings = embed_texts(texts_for_embed, model_name=self.embed_model, use_cache=True)
        title_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        # 4) dense similarities (cosine)
        # similarity between title_emb and each chunk_emb
        dense_sims = []
        for emb in chunk_embs:
            sim = float(np.dot(title_emb, emb))  # embeddings are normalized by SentenceTransformer
            dense_sims.append(sim)

        # 5) sparse BM25 scoring
        try:
            bm25 = BM25(chunks)
            sparse_scores = bm25.score(expanded)
            # bm25.score returns raw scores; normalize them
            sparse_scores = [float(s) for s in sparse_scores]
        except Exception:
            # fallback: zeroed sparse scores
            sparse_scores = [0.0] * len(chunks)

        # 6) normalize both arrays
        dense_norm = normalize_array(np.array(dense_sims))
        sparse_norm = normalize_array(np.array(sparse_scores))

        # 7) combine
        combined = (self.dense_weight * dense_norm) + (self.sparse_weight * sparse_norm)
        # combined array in [0,1] (not guaranteed but usually)
        # pick top-k chunks and aggregate by mean (stable)
        k = min(self.top_k, len(combined))
        topk_idx = np.argsort(combined)[-k:][::-1]  # indices of top k descending
        topk_scores = combined[topk_idx]
        # aggregate final relevance score
        final_score = float(np.mean(topk_scores)) * 100.0

        # best chunk text
        best_idx = int(topk_idx[0]) if len(topk_idx) > 0 else int(np.argmax(combined))
        top_chunk = chunks[best_idx]
        # small truncation for UI
        top_chunk_preview = top_chunk[:800] if top_chunk else ""

        return {
            "score": round(final_score, 2),
            "top_chunk": top_chunk_preview,
            "details": {
                "num_chunks": len(chunks),
                "dense_scores": [float(round(s, 4)) for s in dense_sims],
                "sparse_scores": [float(round(s, 4)) for s in sparse_scores],
                "combined_scores": [float(round(float(s), 4)) for s in combined],
                "topk_indices": [int(i) for i in topk_idx],
            },
        }
