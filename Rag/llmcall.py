import ollama
from utils import log
from config import LLM_MODEL

class LocalLLM:
    def __init__(self):
        self.model = LLM_MODEL
        log(f"Initialized local LLM: {self.model}")
    
    def generate_answer(self, query, context_chunks):
        log("Generating answer with local LLM")
        
        context_with_sources = ""
        for i, chunk in enumerate(context_chunks):
            context_with_sources += f"--- CHUNK {i+1} ---\n{chunk['text']}\n\n"
        
        prompt = f"""You are an expert AI assistant. Using ONLY the context information provided below, answer the user's query. 

USER QUERY: {query}

CONTEXT INFORMATION:
{context_with_sources}

CRITICAL INSTRUCTIONS:
1. You must base your answer ONLY on the provided context
2. If the context contains relevant information, you MUST use it to answer
3. You must base your answer ONLY on the provided context from these specific sources
4. You must cite the actual source URL for each piece of information using this format: (Source: [full URL])
5. If multiple sources support the same point, cite the most relevant one
6. If the context doesn't contain the answer, say "The provided sources don't contain specific information about this."
7. Keep your answer concise and factual
8. Reference the source material where appropriate

Answer:"""
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            answer = response['response']
            log("LLM response generated successfully")
            return answer
        except Exception as e:
            log(f"Error generating answer: {str(e)}", "ERROR")
            return "I encountered an error while generating an answer."
