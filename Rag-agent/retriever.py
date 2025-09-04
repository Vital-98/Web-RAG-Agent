import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from logger import get_logger
from utils import load_json

logger = get_logger("retriever")
EMBED_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
_QUERY_MODEL = None

def _l2norm(v: np.ndarray):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1e-8
    return v / n

def load_index(index_path="data/faiss.index", meta_path="data/meta.json"):
    index = faiss.read_index(index_path)
    meta = load_json(meta_path)
    return index, meta

def _get_query_model():
    global _QUERY_MODEL
    if _QUERY_MODEL is None:
        logger.info(f"RETRIEVER → loading {EMBED_MODEL}")
        _QUERY_MODEL = SentenceTransformer(EMBED_MODEL)
    return _QUERY_MODEL

def embed_query(q: str):
    if not q:
        return np.zeros((1, 1), dtype="float32")
    model = _get_query_model()
    v = model.encode([q], convert_to_numpy=True).astype("float32")
    return _l2norm(v)

def retrieve_topk(query: str, k: int, index, meta):
    if index is None or getattr(index, "ntotal", 0) == 0:
        logger.warning("RETRIEVER → empty index; skipping search")
        return []
    q = embed_query(query)
    if q is None or q.size == 0:
        logger.warning("RETRIEVER → empty query embedding")
        return []
    D, I = index.search(q, k)
    sims, ids = D[0].tolist(), I[0].tolist()
    top = []
    for sim, idx in zip(sims, ids):
        if idx == -1: continue
        m = meta["id_to_meta"][str(idx)] if isinstance(meta["id_to_meta"], dict) else meta["id_to_meta"][idx]
        top.append((idx, float(sim), m))
    logger.info(f"RETRIEVED → {[(t[2]['chunk_id'], t[1]) for t in top]}")
    return top

def build_prompt(query: str, top):
    header = (
        "You are a strict, citation-first assistant. Use ONLY the SOURCE CHUNKS below to answer.\n"
        "For every factual claim append a citation like (Source i).\n"
        "If the answer cannot be determined from the sources, reply exactly: "
        "\"I don't know based on the retrieved sources.\"\n\n"
        f"User question:\n{query}\n\nSOURCE CHUNKS:\n"
    )
    blocks, used = [], {}
    for _id, score, m in top:
        sid = m["source_id"]
        used[sid] = (m["title"], m["url"])
        blocks.append(f"[SOURCE {sid}] {m['title']} | {m['url']} | sim={score:.4f}\n{m['text_preview']}\n")
    src_map = "\n".join([f"Source {s}: {used[s][0]} - {used[s][1]}" for s in sorted(used)])
    tail = (
        "\nDeliver:\n1) A concise answer (<= 250 words) with inline (Source i) citations.\n"
        "2) A SOURCES list mapping Source i -> title + url.\n\nSOURCES:\n" + src_map + "\n"
    )
    return header + "\n".join(blocks) + tail
