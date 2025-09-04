import asyncio
from logger import get_logger
from utils import ensure_dirs
import scraper, embedder
import chunking as chunker
from retriever import load_index, retrieve_topk, build_prompt
from llm_call import generate_with_ollama

logger = get_logger("main")

def pipeline(query: str, k: int = 4, headless: bool = True, refresh: bool = True):
    ensure_dirs()
    logger.info("PIPELINE START → %s", query)

    index_path, meta_path = "data/faiss.index", "data/meta.json"

    if refresh:
        scraped_path = asyncio.run(scraper.run(query, "data/scraped.json", headless=headless))
        chunks_path = chunker.run(scraped_path, "data/chunks.json")
        index_path, meta_path = embedder.run(chunks_path, index_path, meta_path)

    try:
        index, meta = load_index(index_path, meta_path)
    except Exception as e:
        logger.warning(f"LOAD INDEX FAILED → {e}")
        index, meta = None, {"id_to_meta": {}}

    top = retrieve_topk(query, k, index, meta) if query else []
    if not top:
        logger.info("No retrieved chunks; returning fallback message")
        return "I don't know based on the retrieved sources."

    prompt = build_prompt(query, top)
    logger.info("PROMPT chars → %d", len(prompt))
    answer = generate_with_ollama(prompt)
    logger.info("LLM answer len → %d", len(answer))

    print("\n===== FINAL ANSWER =====\n")
    print(answer or "No answer returned.")
    print("\n===== RETRIEVED CHUNKS =====")
    for _id, score, m in top:
        print(f"{m['chunk_id']} (Source {m['source_id']}) sim={score:.4f} url={m['url']}")
    print("\n===== PIPELINE LOGS =====")
    print("Search → Chunk → Embed → Retrieval → Answer (see ./logs)")
    return answer or "I don't know based on the retrieved sources."

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-headless", action="store_true")
    ap.add_argument("--no-refresh", action="store_true", help="Reuse existing index; skip scrape/embed")
    args = ap.parse_args()
    pipeline(args.query, k=args.k, headless=(not args.no_headless), refresh=(not args.no_refresh))
