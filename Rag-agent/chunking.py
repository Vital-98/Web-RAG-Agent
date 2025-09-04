from logger import get_logger
from utils import ensure_dirs, load_json, save_json

logger = get_logger("chunker")

CHUNK_SIZE = 900     # tune if needed
CHUNK_OVERLAP = 150

def _chunk_text(t: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, n, i = [], len(t), 0
    while i < n:
        j = min(i + size, n)
        c = t[i:j].strip()
        if c: chunks.append(c)
        if j == n: break
        i = max(0, j - overlap)
    return chunks

def run(in_path="data/scraped.json", out_path="data/chunks.json"):
    ensure_dirs()
    docs = load_json(in_path)
    out = []
    for d in docs:
        parts = _chunk_text(d["text"])
        for k, c in enumerate(parts, 1):
            out.append({
                "chunk_id": f"S{d['source_id']}-C{k}",
                "source_id": d["source_id"],
                "url": d["url"],
                "title": d["title"],
                "text": c
            })
    save_json(out_path, out)
    logger.info(f"WROTE → {out_path} ({len(out)} chunks) from {len(docs)} docs")
    return out_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/scraped.json")
    ap.add_argument("--out", dest="out", default="data/chunks.json")
    args = ap.parse_args()
    run(args.inp, args.out)
