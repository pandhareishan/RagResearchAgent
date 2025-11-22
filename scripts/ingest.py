import argparse, os, glob, uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

def load_docs(data_dir: str):
    docs = []
    for path in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True):
        if os.path.isfile(path) and any(path.endswith(ext) for ext in [".md", ".txt"]):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append((path, text))
    return docs

def chunk_text(text, chunk_size=700, overlap=100):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sample_docs")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    print(f"Loading docs from {args.data_dir} ...")
    docs = load_docs(args.data_dir)
    if not docs:
        print("No documents found."); return

    model = SentenceTransformer(args.embed_model)
    entries, embeddings = [], []

    for path, text in docs:
        for chunk in chunk_text(text):
            if not chunk.strip(): continue
            entries.append({"path": path, "text": chunk})
            embeddings.append(model.encode(chunk))

    mat = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)

    faiss.write_index(index, os.path.join(args.index_dir, "vectors.faiss"))
    with open(os.path.join(args.index_dir, "entries.json"), "w", encoding="utf-8") as f:
        import json; json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(entries)} chunks from {len(docs)} files into {args.index_dir}")

if __name__ == "__main__":
    main()
