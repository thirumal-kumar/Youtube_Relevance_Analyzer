# core/utils.py
import re
from typing import List

_word_split_re = re.compile(r"\W+")

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = [t for t in _word_split_re.split(text) if t]
    return tokens

def chunk_text(text: str, max_words: int = 50, overlap: float = 0.4) -> List[str]:
    """
    Split text into overlapping chunks of ~max_words words.
    Default smaller chunk size (50) to preserve keywords inside a chunk.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max(1, int(max_words * (1 - overlap)))
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i:i + max_words]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks
