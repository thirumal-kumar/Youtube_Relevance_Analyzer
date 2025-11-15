# core/retrieval.py
from typing import List, Sequence
import math
from collections import Counter, defaultdict

class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75):
        """
        docs: list of documents (strings). We'll tokenize on whitespace.
        """
        self.docs = [self._tok(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        self.k1 = k1
        self.b = b

        self.doc_freqs = []
        self.df = defaultdict(int)
        self.doc_lens = []

        for doc in self.docs:
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            self.doc_lens.append(len(doc))
            for term in freqs.keys():
                self.df[term] += 1

        # precompute IDF per term
        self.idf = {}
        for term, freq in self.df.items():
            # standard BM25 idf smooth
            self.idf[term] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    @staticmethod
    def _tok(text: str):
        if not text:
            return []
        # simple whitespace tokens lowercased
        return [t.lower() for t in text.split() if t]

    def score(self, query: str) -> List[float]:
        """
        Return BM25 score for each doc, in order.
        """
        q_terms = self._tok(query)
        if not q_terms:
            return [0.0] * self.N
        scores = [0.0] * self.N
        for term in q_terms:
            idf = self.idf.get(term, 0.0)
            for i in range(self.N):
                freq = self.doc_freqs[i].get(term, 0)
                denom = freq + self.k1 * (1 - self.b + self.b * (self.doc_lens[i] / max(1.0, self.avgdl)))
                if denom > 0:
                    scores[i] += idf * ((freq * (self.k1 + 1)) / denom)
        return scores

    def topk(self, query: str, k: int = 3):
        sc = self.score(query)
        idxs = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)[:k]
        return [(i, sc[i]) for i in idxs]
