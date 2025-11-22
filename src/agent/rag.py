import os, json
from typing import List, Dict, Tuple
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

class RAGStore:
    def __init__(self, index_dir: str, embed_model: str):
        self.index_dir = index_dir
        self.entries_path = os.path.join(index_dir, "entries.json")
        self.faiss_path = os.path.join(index_dir, "vectors.faiss")
        if not (os.path.exists(self.entries_path) and os.path.exists(self.faiss_path)):
            raise FileNotFoundError(f"Index not found. Run scripts/ingest.py to build it in {index_dir}.")
        self.entries = json.load(open(self.entries_path, "r", encoding="utf-8"))
        self.index = faiss.read_index(self.faiss_path)
        self.model = SentenceTransformer(embed_model)

    def retrieve(self, query: str, k: int = 4) -> List[Dict]:
        q = self.model.encode(query).astype("float32")
        D, I = self.index.search(np.array([q]), k)
        hits = []
        for dist, idx in zip(D[0], I[0]):
            e = self.entries[int(idx)]
            hits.append({"distance": float(dist), "text": e["text"], "path": e["path"]})
        return hits
