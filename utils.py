# small utility helpers used by app and model

import math
import numpy as np
from typing import List

def split_text_into_chunks(text: str, chunk_size_words: int = 120, overlap: int = 20) -> List[str]:
    """
    Simple word-based chunker with overlap.
    """
    words = text.split()
    if not words:
        return []
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

def cosine_sim_matrix(a, b):
    """
    compute cosine similarity between two 2D arrays (n x d) and (m x d)
    returns (n,m) matrix
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)

def normalize_scores(arr):
    import numpy as np
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn = arr.min()
    mx = arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)
