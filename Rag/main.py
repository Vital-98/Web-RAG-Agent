from scraping import GoogleSearcher
from parser import ContentParser
from text_processor import TextProcessor
from llmcall import LocalLLM
from utils import log, format_results
from config import SEARCH_RESULTS_COUNT, TOP_K_URLS, TOP_K_CHUNKS, CHUNK_SIZE, CHUNK_OVERLAP

def main():
    user_query = input("Enter your query: ")
    
    log("=" * 60)
    log("Starting RAG Pipeline")
    log("=" * 60)
    
    # Initialize components
    searcher = GoogleSearcher()
    parser = ContentParser()
    processor = TextProcessor()
    llm = LocalLLM()
    
    try:
        # Stage 1: Search
        log("\n1. 🔍 WEB SEARCH STAGE")
        search_results = searcher.search(user_query, SEARCH_RESULTS_COUNT)
        log(f"Search results:\n{format_results(search_results)}")
        
        # Stage 2: Content Retrieval & Parsing
        log("\n2. 📥 CONTENT RETRIEVAL STAGE")
        all_chunks = []
        sources = []
        
        for i, result in enumerate(search_results[:TOP_K_URLS]):
            content = parser.fetch_and_clean(result['url'])
            if content:
                chunks = processor.chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
                all_chunks.extend(chunks)
                sources.extend([result['url']] * len(chunks))
                log(f"URL {i+1}: {len(chunks)} chunks")
        
        # Stage 3: Processing & Ranking
        log("\n3. 🧮 PROCESSING & RANKING STAGE")
        if not all_chunks:
            log("No content retrieved. Cannot generate answer.", "ERROR")
            return
            
        ranked_chunks = processor.rank_chunks(user_query, all_chunks, TOP_K_CHUNKS)
        
        # Add source URLs to ranked chunks
        for chunk in ranked_chunks:
            chunk['source'] = sources[chunk['index']]
        
        log("Top ranked chunks:")
        for i, chunk in enumerate(ranked_chunks):
            log(f"Chunk {i+1} (Score: {chunk['score']:.3f}): {chunk['text'][:100]}...")
        
        # Stage 4: Answer Generation
        log("\n4. 🤖 ANSWER GENERATION STAGE")
        answer = llm.generate_answer(user_query, ranked_chunks)
        
        log("\n5. ✅ FINAL ANSWER")
        log("=" * 40)
        print(f"\n{answer}")
        log("=" * 40)
        
    finally:
        searcher.close()

if __name__ == "__main__":
    main()