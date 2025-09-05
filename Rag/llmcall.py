import ollama
from utils import log
from config import LLM_MODEL

class LocalLLM:
    def __init__(self):
        self.model = LLM_MODEL
        log(f"Initialized local LLM: {self.model}")
    
    def generate_answer(self, query, context_chunks):
        log("Generating answer with local LLM")
        
        # Prepare context with sources
        context_with_sources = ""
        for i, chunk in enumerate(context_chunks):
            context_with_sources += f"--- CHUNK {i+1} ---\n{chunk['text']}\n\n"
        
        # Create prompt that forces citation
        prompt = f"""Based ONLY on the following context information, answer the user's query. 
Your answer must be concise, accurate, and must ONLY be based on the provided context. 
If the context does not contain the answer, say "Based on the sources I found, I cannot answer this question."

USER QUERY: {query}

CONTEXT INFORMATION:
{context_with_sources}

IMPORTANT: Always cite your source by referring to the chunk number at the end of relevant sentences, like this: (Source: Chunk X).
Answer:"""
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            answer = response['response']
            log("LLM response generated successfully")
            return answer
        except Exception as e:
            log(f"Error generating answer: {str(e)}", "ERROR")
            return "I encountered an error while generating an answer."