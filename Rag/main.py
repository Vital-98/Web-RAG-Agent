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
    
    
    searcher = GoogleSearcher()
    parser = ContentParser()
    processor = TextProcessor()
    llm = LocalLLM()
    
    try:
        # Search
        log("\n1.WEB SCRAPING")
        search_results = searcher.search(user_query, SEARCH_RESULTS_COUNT)
        log(f"Search results:\n{format_results(search_results)}")
        
        # Content Parsing
        log("\n2.CONTENT PARSING")
        all_chunks = []
        sources = []
        
        for i, result in enumerate(search_results[:TOP_K_URLS]):
            content = parser.fetch_and_clean(result['url'])
            if content:
                chunks = processor.chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
                all_chunks.extend(chunks)
                sources.extend([result['url']] * len(chunks))
                log(f"URL {i+1}: {len(chunks)} chunks")
        
        # Processing & Ranking
        log("\n3.EMBEDDING & RANKING")
        if not all_chunks:
            log("No content retrieved. Cannot generate answer.", "ERROR")
            return
            
        ranked_chunks = processor.rank_chunks(user_query, all_chunks, TOP_K_CHUNKS)
        
        #url to ranked chunks
        for chunk in ranked_chunks:
            chunk['source'] = sources[chunk['index']]
        
        log("Top ranked chunks:")
        for i, chunk in enumerate(ranked_chunks):
            log(f"Chunk {i+1} (Score: {chunk['score']:.3f}) from: {chunk['source']}")
            log(f"Content: {chunk['text'][:100]}...")
        # generate answer
        log("\n4. RESPONSE GENERATION")
        answer = llm.generate_answer(user_query, ranked_chunks)
        
        log("\n5.FINAL ANSWER")
        log("=" * 40)
        print(f"\n{answer}")
        log("=" * 40)
        
    finally:
        searcher.close()

if __name__ == "__main__":
    main()
