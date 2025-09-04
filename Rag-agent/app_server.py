from fastapi import FastAPI
from pydantic import BaseModel
from logger import get_logger
from utils import ensure_dirs
from retriever import load_index, retrieve_topk, build_prompt
from llm_call import generate_with_ollama
import embedder
import chunking as chunker
import asyncio
import scraper

logger = get_logger("api")
app = FastAPI(title="RAG Demo API")

class AskRequest(BaseModel):
    query: str
    k: int = 4
    refresh: bool = False
    headless: bool = True

@app.on_event("startup")
async def _startup():
    ensure_dirs()
    logger.info("API STARTED")

@app.post("/ask")
async def ask(req: AskRequest):
    ensure_dirs()
    index_path, meta_path = "data/faiss.index", "data/meta.json"

    if req.refresh:
        scraped_path = await scraper.run(req.query, "data/scraped.json", headless=req.headless)
        chunks_path = chunker.run(scraped_path, "data/chunks.json")
        embedder.run(chunks_path, index_path, meta_path)

    try:
        index, meta = load_index(index_path, meta_path)
    except Exception as e:
        logger.warning(f"LOAD INDEX FAILED → {e}")
        index, meta = None, {"id_to_meta": {}}

    top = retrieve_topk(req.query, req.k, index, meta) if req.query else []
    if not top:
        return {"answer": "I don't know based on the retrieved sources.", "chunks": []}

    prompt = build_prompt(req.query, top)
    answer = generate_with_ollama(prompt)
    chunks = [
        {"id": midx, "score": score, "meta": meta}
        for (midx, score, meta) in top
    ]
    return {"answer": answer or "I don't know based on the retrieved sources.", "chunks": chunks}


