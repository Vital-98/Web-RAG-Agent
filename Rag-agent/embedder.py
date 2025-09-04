import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from logger import get_logger
from utils import ensure_dirs, load_json, save_json

logger = get_logger("embedder")
EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"  # tuned for QA retrieval

def _l2norm(v: np.ndarray):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1e-8
    return v / n

def run(chunks_path="data/chunks.json", index_path="data/faiss.index", meta_path="data/meta.json"):
    ensure_dirs()
    chunks = load_json(chunks_path)
    if not chunks:
        logger.warning("EMBED → no chunks to embed; skipping index build")
        save_json(meta_path, {"faiss_dim": 0, "id_to_meta": {}, "embed_model": EMBED_MODEL})
        faiss.write_index(faiss.IndexFlatIP(1), index_path)
        return index_path, meta_path

    texts = [c["text"] for c in chunks if c.get("text")]
    if not texts:
        logger.warning("EMBED → all chunks empty; skipping index build")
        save_json(meta_path, {"faiss_dim": 0, "id_to_meta": {}, "embed_model": EMBED_MODEL})
        faiss.write_index(faiss.IndexFlatIP(1), index_path)
        return index_path, meta_path

    logger.info(f"EMBED → loading {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    embs = _l2norm(embs)
    dim = embs.shape[1]
    logger.info(f"EMBED → {embs.shape}")

    index = faiss.IndexFlatIP(dim)    # IP on normalized = cosine
    if embs.size > 0:
        index.add(embs)
    faiss.write_index(index, index_path)

    id_to_meta = {i: {
        "chunk_id": c["chunk_id"],
        "source_id": c["source_id"],
        "title": c["title"],
        "url": c["url"],
        "text_preview": c["text"][:500]
    } for i, c in enumerate(chunks)}

    save_json(meta_path, {"faiss_dim": dim, "id_to_meta": id_to_meta, "embed_model": EMBED_MODEL})
    logger.info(f"WROTE → {index_path} ({index.ntotal} vecs), {meta_path}")
    return index_path, meta_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks.json")
    ap.add_argument("--index", default="data/faiss.index")
    ap.add_argument("--meta", default="data/meta.json")
    args = ap.parse_args()
    run(args.chunks, args.index, args.meta)
