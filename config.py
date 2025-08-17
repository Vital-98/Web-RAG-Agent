import os
from typing import Dict, Any

class RAGConfig:
    """for RAG Agent"""
    

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    
    # Model settings (Gemma 3N E2B)
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL")
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    

    DEFAULT_SEARCH_ENGINE = os.getenv("DEFAULT_SEARCH_ENGINE", "google")
    DEFAULT_MAX_RESULTS = int(os.getenv("DEFAULT_MAX_RESULTS", "3"))
    
    # Fallback search settings
    ENABLE_SEARCH_FALLBACK = os.getenv("ENABLE_SEARCH_FALLBACK", "true").lower() == "true"
    FALLBACK_THRESHOLD = int(os.getenv("FALLBACK_THRESHOLD", "2"))  # Minimum results before fallback
    

    
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "3000"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
    

    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))  # Increased for parallel processing
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    
    # Parallel processing settings
    ENABLE_PARALLEL_EXTRACTION = os.getenv("ENABLE_PARALLEL_EXTRACTION", "true").lower() == "true"
    ENABLE_PARALLEL_PROCESSING = os.getenv("ENABLE_PARALLEL_PROCESSING", "true").lower() == "true"
    

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    MAIN_PROMPT_TEMPLATE = """Context: {context}

User Query: {query}

Please provide a comprehensive, accurate, and concise answer based on the context provided. If the context doesn't contain enough information to answer the query, please say so clearly.

Answer:"""
    
    SUMMARIZATION_PROMPT = """Please provide a concise summary of the following content in 2-3 sentences:

{content}

Summary:"""
    
    KEY_POINTS_PROMPT = """Please extract 3 key points from the following content. Return them as a simple list:

{content}

Key points:"""
    

    ERROR_MESSAGES = {
        "ollama_connection": "Failed to connect to Ollama. Make sure it's running: ollama serve",
        "model_not_found": "Model not found. Install with: ollama pull gemma-3n-E2B-it",
        "search_failed": "Search failed. Please try again or use a different search engine.",
        "no_results": "No relevant information found. Please try rephrasing your query.",
        "fallback_failed": "All search engines failed. Please check your internet connection."
    }

    # Timeout settings for different operations
    TIMEOUTS = {
        "search": 15,
        "extraction": 20,
        "llm_generation": 30,
        "connection": 5
    }
    
    # Retry settings
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get configuration for Gemma 3N model"""
        return {
            "model_name": cls.DEFAULT_MODEL,
            "base_url": cls.OLLAMA_BASE_URL,
            "timeout": cls.OLLAMA_TIMEOUT,
            "max_tokens": cls.DEFAULT_MAX_TOKENS,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "top_p": cls.DEFAULT_TOP_P
        }
    
    @classmethod
    def get_processing_config(cls) -> Dict[str, Any]:
        """Get processing configuration"""
        return {
            "max_workers": cls.MAX_WORKERS,
            "enable_parallel_extraction": cls.ENABLE_PARALLEL_EXTRACTION,
            "enable_parallel_processing": cls.ENABLE_PARALLEL_PROCESSING,
            "enable_caching": cls.ENABLE_CACHING,
            "max_cache_size": cls.MAX_CACHE_SIZE
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "default_engine": cls.DEFAULT_SEARCH_ENGINE,
            "max_results": cls.DEFAULT_MAX_RESULTS,
            "enable_fallback": cls.ENABLE_SEARCH_FALLBACK,
            "fallback_threshold": cls.FALLBACK_THRESHOLD,
            "timeout": cls.TIMEOUTS["search"]
        }
# Default configuration instance
config = RAGConfig()

