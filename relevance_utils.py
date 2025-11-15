# relevance_utils.py
"""
Helper utilities for the Hybrid RAG Relevance Model:
- chunk_text_semantic: word-based overlapping chunker
- expand_title: lightweight title -> topic expansion (keyword heuristics)
- embed_texts, cosine_sim_matrix: embedding helpers with caching
- normalize_array: simple min-max normalization
"""

from typing import List, Tuple
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import os
import pickle

# small stopword set (keep local, no external libs)
_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","is","are","this",
    "that","it","as","be","we","you","your","our","from","at","have","has","but","not",
    "will","can","should","they","their","i","me","my","so","do"
}

# Embedding cache directory (optional, speeds repeated runs)
_EMB_CACHE_DIR = ".emb_cache"
os.makedirs(_EMB_CACHE_DIR, exist_ok=True)

def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def chunk_text_semantic(text: str, chunk_size_words: int = 160, overlap: int = 30) -> List[str]:
    """
    Word-based overlapping chunker.
    Default chunk size ~160 words with 30-word overlap.
    """
    if not text or not text.strip():
        return []
    words = text.strip().split()
    if len(words) <= chunk_size_words:
        return [" ".join(words)]
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = min(i + chunk_size_words, n)
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        if j == n:
            break
        i = j - overlap
    return chunks

def expand_title(title: str) -> str:
    """
    Lightweight title expansion to produce a richer 'topic string'.
    Strategy:
      - Keep original title
      - Extract keywords (remove stopwords, punctuation)
      - Create a short paraphrase-like phrase combining keywords
    This is NOT an LLM; it's a deterministic augmentation that stabilizes embeddings.
    """
    if not title or not title.strip():
        return ""
    # clean punctuation
    cleaned = re.sub(r"[^\w\s]", " ", title).lower()
    words = [w for w in cleaned.split() if w and w not in _STOPWORDS]
    # keep top up to 10 words
    keywords = words[:10]
    phrase = " ".join(keywords)
    # common paraphrase adder
    extras = []
    if "postman" in cleaned:
        extras.append("postman alternative api client")
    if "api" in cleaned:
        extras.append("api testing client")
    # combine original + keyword phrase + extras
    pieces = [title.strip(), phrase] + extras
    # deduplicate while preserving order
    seen = set()
    out = []
    for p in pieces:
        p = p.strip()
        if p and p.lower() not in seen:
            out.append(p)
            seen.add(p.lower())
    return " . ".join(out)

# ---------- Embedding helpers with caching ----------
_DEFAULT_EMBEDDER = None

def get_default_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = SentenceTransformer(model_name)
    return _DEFAULT_EMBEDDER

def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2", use_cache: bool = True) -> List[np.ndarray]:
    """
    Return list of numpy embeddings. Uses disk cache per-text for speed on repeated runs.
    """
    if not texts:
        return []
    embedder = get_default_embedder(model_name)
    embeddings = []
    to_compute = []
    compute_indices = []
    for i, t in enumerate(texts):
        key = _hash_text(t)
        cache_file = os.path.join(_EMB_CACHE_DIR, f"{model_name.replace('/', '_')}_{key}.pkl")
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    emb = pickle.load(f)
                embeddings.append(emb)
                continue
            except Exception:
                pass
        # placeholder
        embeddings.append(None)
        to_compute.append(t)
        compute_indices.append(i)

    if to_compute:
        computed = embedder.encode(to_compute, convert_to_numpy=True, normalize_embeddings=True)
        for idx_in_list, emb in enumerate(computed):
            i = compute_indices[idx_in_list]
            embeddings[i] = emb
            # write cache
            key = _hash_text(texts[i])
            cache_file = os.path.join(_EMB_CACHE_DIR, f"{model_name.replace('/', '_')}_{key}.pkl")
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(emb, f)
            except Exception:
                pass

    return embeddings

def cosine_sim_matrix(a, b):
    """
    compute cosine similarity between two 2D arrays (n x d) and (m x d)
    returns (n,m) matrix
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)

def normalize_array(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
