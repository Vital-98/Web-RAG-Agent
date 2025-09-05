from sentence_transformers import SentenceTransformer, util
import numpy as np
from utils import log
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

class TextProcessor:
    def __init__(self):
        log(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    def chunk_text(self, text, chunk_size=500, chunk_overlap=50):
        log("Splitting text into chunks")
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
                
        log(f"Created {len(chunks)} chunks")
        return chunks
    
    def rank_chunks(self, query, chunks, top_k=3):
        log("Ranking chunks by relevance to query")
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        chunk_embeddings = self.embedder.encode(chunks, convert_to_tensor=True)
        
        # i found that cosine similarity was oftten mentioned while ranking used in raf flows
        cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
        
        top_results = cos_scores.topk(k=min(top_k, len(cos_scores)))
        ranked_chunks = []
        for score, idx in zip(top_results[0], top_results[1]):
            ranked_chunks.append({
                "text": chunks[idx],
                "score": score.item(),
                "index": idx.item()
            })
            
        log(f"Top chunk score: {ranked_chunks[0]['score']:.3f}")

        return ranked_chunks
