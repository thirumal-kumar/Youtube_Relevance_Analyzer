# model.py (improved)
import numpy as np
from typing import Dict, Any, List
from relevance_utils import (
    chunk_text_semantic,
    embed_texts,
    normalize_array,
)
from retrieval import BM25
from collections import Counter
import math
import re

# small stopword set reused (kept local)
_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","this",
    "that","it","as","be","we","you","your","our","from","at","have","has","but","not",
    "will","can","should","they","their","i","me","my","so","do","how","what","why",
    "which","when","where","here","there"
}

def _clean_and_tokenize(text: str):
    if not text:
        return []
    text = re.sub(r"[^\w\s]", " ", text).lower()
    toks = [t for t in text.split() if t and t not in _STOPWORDS]
    return toks

def _top_keywords_from_texts(texts: List[str], top_n: int = 8):
    """
    Simple TF-based keyword extraction across a list of texts.
    Returns top_n keywords by frequency (excluding stopwords).
    """
    ctr = Counter()
    for t in texts:
        toks = _clean_and_tokenize(t)
        ctr.update(toks)
    if not ctr:
        return []
    return [w for w, _ in ctr.most_common(top_n)]

def _rrf_score_from_rank_pairs(rank_dense: np.ndarray, rank_sparse: np.ndarray, k: float = 60.0):
    """
    Compute Reciprocal Rank Fusion (RRF) combined score from ranks.
    rank_* are integer arrays (1-based ranks), shape (n_chunks,).
    Returns array of floats (higher = better).
    """
    # ensure numpy arrays
    rd = np.asarray(rank_dense, dtype=float)
    rs = np.asarray(rank_sparse, dtype=float)
    # RRF contribution from each rank source
    return 1.0 / (k + rd) + 1.0 / (k + rs)

def _ranks_from_scores(scores: np.ndarray, descending: bool = True):
    """
    Convert scores -> 1-based ranks. Ties handled by stable sorting.
    Highest score -> rank 1 if descending True.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=int)
    if descending:
        order = np.argsort(-arr, kind="stable")
    else:
        order = np.argsort(arr, kind="stable")
    ranks = np.empty_like(order, dtype=int)
    # assign 1-based ranks
    for pos, idx in enumerate(order):
        ranks[idx] = pos + 1
    return ranks

class RelevanceModel:
    """
    Improved Hybrid RAG Relevance Model:
      - Default embedder: intfloat/e5-large-v2 (better semantic retrieval)
      - Smaller chunk size (80 words default), overlap 20
      - Title expansion via embedding-driven keyword extraction (no LLM)
      - Dense similarity (cosine) + Sparse BM25
      - Reciprocal Rank Fusion (RRF) to combine ranks
      - Aggregation: median(top_k) -> final score scaled 0-100
    """

    def __init__(
        self,
        embed_model: str = "intfloat/e5-large-v2",
        chunk_size_words: int = 80,
        overlap: int = 20,
        top_k: int = 3,
        rrf_k: float = 60.0,
        use_cache: bool = True,
    ):
        self.embed_model = embed_model
        self.chunk_size_words = int(chunk_size_words)
        self.overlap = int(overlap)
        self.top_k = int(top_k)
        self.rrf_k = float(rrf_k)
        self.use_cache = bool(use_cache)

    def compute_relevance(self, title: str, transcript: str) -> Dict[str, Any]:
        """
        Compute relevance between `title` (expected topic) and `transcript` (video transcript).
        Returns a dict with:
          - score: float (0-100)
          - top_chunk: str
          - details: dense_scores, sparse_scores, rrf_scores, ranks, num_chunks, topk_indices
        """
        # Basic validation
        if not title or not title.strip():
            return {"score": 0.0, "reason": "empty title", "top_chunk": ""}
        if not transcript or not transcript.strip():
            return {"score": 0.0, "reason": "empty transcript", "top_chunk": ""}

        # 1) chunk transcript (more granular)
        chunks = chunk_text_semantic(transcript, chunk_size_words=self.chunk_size_words, overlap=self.overlap)
        if not chunks:
            return {"score": 0.0, "reason": "no chunks produced", "top_chunk": ""}

        n_chunks = len(chunks)

        # 2) initial embeddings to find top matching chunks (for title expansion)
        # We'll embed title + chunks together
        try:
            texts_for_embed = [title] + chunks
            embeddings = embed_texts(texts_for_embed, model_name=self.embed_model, use_cache=self.use_cache)
            if not embeddings or len(embeddings) < (1 + n_chunks):
                # fallback: try without cache
                embeddings = embed_texts(texts_for_embed, model_name=self.embed_model, use_cache=False)
        except Exception as e:
            # If embedder fails, return safe fallback
            return {"score": 0.0, "reason": f"embedding failure: {e}", "top_chunk": ""}

        title_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        # 3) dense similarities (cosine) between title and chunk embeddings
        # embeddings are expected to be normalized by embedder (E5 does not always normalize),
        # so we do a safe cosine computation:
        def _cosine(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
            return float(np.dot(a, b) / denom)

        dense_scores = np.array([_cosine(title_emb, ce) for ce in chunk_embs], dtype=float)

        # 4) title expansion: pick top matching chunks (by dense score), extract keywords and enrich title
        try:
            top_chunk_count_for_expansion = min(max(1, n_chunks // 8), 6)  # heuristic
            top_chunk_indices = np.argsort(dense_scores)[-top_chunk_count_for_expansion:][::-1]
            top_texts = [chunks[i] for i in top_chunk_indices]
            expansion_keywords = _top_keywords_from_texts(top_texts, top_n=10)
            # form expanded title (original + keywords)
            expanded_title = " . ".join([title.strip()] + expansion_keywords)
        except Exception:
            expanded_title = title.strip()

        # 5) sparse BM25 scoring (use expanded title)
        try:
            bm25 = BM25(chunks)
            sparse_scores_raw = bm25.score(expanded_title)
            sparse_scores = np.array([float(s) for s in sparse_scores_raw], dtype=float)
        except Exception:
            sparse_scores = np.zeros(n_chunks, dtype=float)

        # 6) Robust normalization: convert dense & sparse to [0,1] with safe handling
        # If array has no variance, give small epsilon differences to avoid equal ranks
        def _safe_normalize(arr: np.ndarray):
            if arr.size == 0:
                return arr
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            if mx == mn:
                # try zscore-like tiny noise to preserve order
                return np.linspace(0, 1, num=arr.size)
            return (arr - mn) / (mx - mn)

        dense_norm = _safe_normalize(dense_scores)
        sparse_norm = _safe_normalize(sparse_scores)

        # 7) compute ranks from normalized scores (1 = best)
        rank_dense = _ranks_from_scores(dense_norm, descending=True)
        rank_sparse = _ranks_from_scores(sparse_norm, descending=True)

        # 8) RRF combination (uses ranks — robust across scoring scales)
        rrf_scores = _rrf_score_from_rank_pairs(rank_dense, rank_sparse, k=self.rrf_k)

        # For diagnostics, also compute a fallback weighted combined score (normalized)
        weighted_combined = (0.6 * dense_norm) + (0.4 * sparse_norm)
        weighted_norm = _safe_normalize(weighted_combined)

        # 9) final selection: choose top_k indices using RRF ranking
        k = min(self.top_k, n_chunks)
        topk_idx = np.argsort(rrf_scores)[-k:][::-1]  # indices of top k by RRF (descending)
        topk_scores_rrf = rrf_scores[topk_idx]

        # Aggregate final score as median of chosen RRF scores, then scale to 0-100
        if topk_scores_rrf.size == 0:
            final_score_pct = 0.0
        else:
            final_score_pct = float(np.median(topk_scores_rrf))  # rrf values are small; we'll rescale below

            # rescale RRF to 0-100 using normalization across all RRF scores
            # if all zero, fallback to weighted norm median
            if np.allclose(rrf_scores, 0.0):
                final_score_pct = float(np.median(weighted_norm))
            else:
                # normalize rrf_scores to [0,1]
                rrf_norm = _safe_normalize(rrf_scores)
                final_score_pct = float(np.median(rrf_norm[topk_idx]))

            final_score_pct = max(0.0, min(1.0, final_score_pct)) * 100.0

        # 10) pick representative top chunk text
        if len(topk_idx) > 0:
            best_idx = int(topk_idx[0])
        else:
            best_idx = int(np.argmax(rrf_scores) if rrf_scores.size else 0)
        top_chunk = chunks[best_idx] if 0 <= best_idx < len(chunks) else ""
        top_chunk_preview = top_chunk[:800] if top_chunk else ""

        # 11) prepare debug details (rounded for readability)
        def _round_list(a):
            return [float(round(float(x), 6)) for x in (a.tolist() if isinstance(a, np.ndarray) else a)]

        details = {
            "num_chunks": n_chunks,
            "dense_scores": _round_list(dense_scores),
            "sparse_scores": _round_list(sparse_scores),
            "dense_norm": _round_list(dense_norm),
            "sparse_norm": _round_list(sparse_norm),
            "rank_dense": [int(r) for r in rank_dense.tolist()] if rank_dense.size else [],
            "rank_sparse": [int(r) for r in rank_sparse.tolist()] if rank_sparse.size else [],
            "rrf_scores": _round_list(rrf_scores),
            "weighted_combined": _round_list(weighted_combined),
            "topk_indices": [int(i) for i in topk_idx],
            "expanded_title": expanded_title,
            "expansion_keywords": expansion_keywords if 'expansion_keywords' in locals() else [],
        }

        return {
            "score": round(final_score_pct, 2),
            "top_chunk": top_chunk_preview,
            "details": details,
        }
