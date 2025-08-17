import requests
import re
import json
import time
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import hashlib
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import search libraries
try:
    from googlesearch import search
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("Warning: googlesearch-python not available. Install with: pip install googlesearch-python")

try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    print("Warning: duckduckgo-search not available. Install with: pip install duckduckgo-search")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    relevance_score: float = 0.0
    source_engine: str = "unknown"

@dataclass
class ProcessedDocument:
    """Represents a processed document with extracted information"""
    url: str
    title: str
    content: str
    summary: str
    key_points: List[str]
    source_credibility: float
    source_engine: str

class OllamaLLM:
    """Local LLM implementation using Ollama"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            logger.info(f"✓ Connected to Ollama at {self.base_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"X Failed to connect to Ollama at {self.base_url}")
            logger.error(f"Make sure Ollama is running: ollama serve")
            raise ConnectionError(f"Cannot connect to Ollama: {e}")
    
    def _check_model_availability(self):
        """Check if the specified model is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            
            available_models = [model['name'] for model in models]
            if self.model_name not in available_models:
                logger.warning(f"Model '{self.model_name}' not found. Available models: {available_models}")
                logger.info(f"To install: ollama pull {self.model_name}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 1000) -> str:
        """Generate response using Ollama"""
        try:
            # Check model availability
            if not self._check_model_availability():
                return "Error: Model not available. Please install the model first."
            
            # Prepare the full prompt with context
            if context:
                full_prompt = f"""Context: {context}

User Query: {prompt}

Please provide a comprehensive, accurate, and concise answer based on the context provided. If the context doesn't contain enough information to answer the query, please say so clearly.

Answer:"""
            else:
                full_prompt = f"""User Query: {prompt}

Please provide a helpful and accurate answer to this query.

Answer:"""
            
            # Make request to Ollama
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 1.0,  # Official recommended setting
                    "top_k": 64,         # Official recommended setting
                    "top_p": 0.95,       # Official recommended setting
                    "min_p": 0.0,        # Official recommended setting
                    "repeat_penalty": 1.0 # Disabled repetition penalty
                }
            }
            
            response = self.session.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', 'No response generated')
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return f"Error: Failed to generate response from Ollama: {e}"
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generation: {e}")
            return f"Error: Unexpected error occurred: {e}"
    
    def summarize_content(self, content: str) -> str:
        """Summarize content using Ollama"""
        if not content:
            return ""
        
        prompt = f"""Please provide a concise summary of the following content in 2-3 sentences:

{content[:3000]}

Summary:"""
        
        try:
            return self.generate_response(prompt, max_tokens=200)
        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            # Fallback to simple summarization
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            if len(sentences) <= 3:
                return content
            return '. '.join(sentences[:3]) + '.'
    
    def extract_key_points(self, content: str) -> List[str]:
        """Extract key points using Ollama"""
        if not content:
            return []
        
        prompt = f"""Please extract 3 key points from the following content. Return them as a simple list:

{content[:2000]}

Key points:"""
        
        try:
            response = self.generate_response(prompt, max_tokens=300)
            # Parse the response to extract key points
            lines = response.split('\n')
            key_points = []
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    key_points.append(line[1:].strip())
                elif line and line[0].isdigit() and '.' in line:
                    key_points.append(line.split('.', 1)[1].strip())
            
            return key_points[:3] if key_points else []
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []

class WebSearcher:
    """Handles web search operations with fallback support"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.search_engines = {
            'google': self._search_google,
            'duckduckgo': self._search_duckduckgo
        }
        self.engine_availability = self._check_engine_availability()
    
    def _check_engine_availability(self) -> Dict[str, bool]:
        """Check which search engines are available"""
        availability = {}
        availability['google'] = GOOGLE_AVAILABLE
        availability['duckduckgo'] = DUCKDUCKGO_AVAILABLE
        
        logger.info(f"Search engine availability: {availability}")
        return availability
    
    def search_with_fallback(self, query: str, preferred_engine: str = 'google', max_results: int = 5) -> List[SearchResult]:
        """Search with automatic fallback to other engines"""
        all_results = []
        
        # Try preferred engine first
        if self.engine_availability.get(preferred_engine, False):
            try:
                logger.info(f"Searching with {preferred_engine}...")
                results = self.search(query, preferred_engine, max_results)
                all_results.extend(results)
                logger.info(f"✓ Found {len(results)} results from {preferred_engine}")
            except Exception as e:
                logger.warning(f"✗ {preferred_engine} search failed: {e}")
        
        # If not enough results, try other engines
        if len(all_results) < max_results // 2:
            for engine, is_available in self.engine_availability.items():
                if engine != preferred_engine and is_available:
                    try:
                        logger.info(f"Trying fallback search with {engine}...")
                        remaining_results = max_results - len(all_results)
                        results = self.search(query, engine, remaining_results)
                        all_results.extend(results)
                        logger.info(f"✓ Found {len(results)} additional results from {engine}")
                        
                        if len(all_results) >= max_results:
                            break
                    except Exception as e:
                        logger.warning(f"✗ {engine} fallback search failed: {e}")
        
        # Remove duplicates and filter out problematic URLs
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result.url not in seen_urls and self._is_valid_url(result.url):
                unique_results.append(result)
                seen_urls.add(result.url)
        
        logger.info(f"Total valid unique results: {len(unique_results)}")
        return unique_results[:max_results]
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and not problematic"""
        try:
            # Check if URL has proper scheme
            if not url.startswith(('http://', 'https://')):
                return False
            
            # Parse URL to check structure
            parsed = urlparse(url)
            
            # Check for common problematic patterns
            problematic_patterns = [
                '/search?',  # Search result pages
                '/results?',  # Result pages
                'google.com/search',  # Google search pages
                'bing.com/search',  # Bing search pages
                'duckduckgo.com/',  # DuckDuckGo search pages
                'youtube.com/results',  # YouTube results
                'amazon.com/s?',  # Amazon search results
            ]
            
            for pattern in problematic_patterns:
                if pattern in url.lower():
                    logger.debug(f"Filtering out problematic URL: {url}")
                    return False
            
            # Check if domain is valid
            if not parsed.netloc or len(parsed.netloc) < 3:
                return False
            
            return True
            
        except Exception:
            return False
    
    def search(self, query: str, engine: str = 'google', max_results: int = 5) -> List[SearchResult]:
        """Search the web using the specified search engine"""
        if engine not in self.search_engines:
            raise ValueError(f"Unsupported search engine: {engine}")
        
        if not self.engine_availability.get(engine, False):
            raise ValueError(f"Search engine {engine} is not available")
        
        try:
            results = self.search_engines[engine](query, max_results)
            for result in results:
                result.source_engine = engine
            return results
        except Exception as e:
            logger.error(f"Search failed for {engine}: {e}")
            return []
    
    def _search_google(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using Google"""
        results = []
        try:
            search_results = search(query, num_results=max_results, lang="en")
            for url in search_results:
                # Validate URL format
                if not url.startswith(('http://', 'https://')):
                    logger.warning(f"Skipping invalid URL: {url}")
                    continue
                
                try:
                    parsed_url = urlparse(url)
                    domain = parsed_url.netloc
                    title = domain.replace('www.', '')
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=f"Result from {domain}",
                        source_engine="google"
                    ))
                except Exception as url_error:
                    logger.warning(f"Error parsing URL {url}: {url_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Google search error: {e}")
        
        return results
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        results = []
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                for result in search_results:
                    results.append(SearchResult(
                        title=result.get('title', ''),
                        url=result.get('link', ''),
                        snippet=result.get('body', ''),
                        source_engine="duckduckgo"
                    ))
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
        
        return results

class ContentExtractor:
    """Extracts and processes content from web pages with parallel processing"""
    
    def __init__(self, max_workers: int = 4):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Updated headers to be more realistic and avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.session.timeout = 15
        self.max_workers = max_workers
    
    def extract_content_parallel(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """Extract content from multiple URLs in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all extraction tasks
            future_to_url = {
                executor.submit(self.extract_content, url): url 
                for url in urls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    results[url] = content
                    if content:
                        logger.info(f"✓ Extracted content from {url}")
                    else:
                        logger.warning(f"✗ Failed to extract content from {url}")
                except Exception as e:
                    logger.error(f"✗ Error extracting content from {url}: {e}")
                    results[url] = None
        
        return results
    
    def extract_content(self, url: str) -> Optional[str]:
        """Extract main content from a web page"""
        try:
            # Validate URL before making request
            if not url.startswith(('http://', 'https://')):
                logger.warning(f"Invalid URL format: {url}")
                return None
            
            # Add more realistic headers to avoid 403 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Check if content is HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"Non-HTML content from {url}: {content_type}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try to find main content area
            main_content = None
            
            # Look for common content selectors
            selectors = [
                'main', 'article', '.content', '.post-content', 
                '.entry-content', '#content', '.main-content',
                '.post-body', '.article-content', '.story-content',
                '.page-content', '.text-content', '.body-content'
            ]
            
            for selector in selectors:
                main_content = soup.select_one(selector)
                if main_content and len(main_content.get_text().strip()) > 100:
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.body or soup
            
            # Extract text content
            text = main_content.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Check if we got meaningful content
            if len(text.strip()) < 50:
                logger.warning(f"Too little content extracted from {url}: {len(text)} characters")
                return None
            
            return text[:3000]  # Limit content length for Ollama
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.warning(f"Access forbidden (403) for {url} - site may block scrapers")
            elif e.response.status_code == 404:
                logger.warning(f"Page not found (404) for {url}")
            else:
                logger.error(f"HTTP error {e.response.status_code} for {url}: {e}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout while fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return None
    
    def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract metadata from a web page"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                'title': soup.title.string if soup.title else '',
                'description': '',
                'keywords': '',
                'author': ''
            }
            
            # Extract meta tags
            meta_tags = soup.find_all('meta')
            for tag in meta_tags:
                name = tag.get('name', '').lower()
                property_attr = tag.get('property', '').lower()
                content = tag.get('content', '')
                
                if name == 'description' or property_attr == 'og:description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = content
                elif name == 'author' or property_attr == 'article:author':
                    metadata['author'] = content
                elif property_attr == 'og:title' and not metadata['title']:
                    metadata['title'] = content
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {url}: {e}")
            return {}

class RAGAgent:
    """Main RAG agent with fallback search and parallel processing"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", max_workers: int = 4):
        self.searcher = WebSearcher()
        self.extractor = ContentExtractor(max_workers=max_workers)
        self.llm = OllamaLLM(model_name, base_url)
        self.cache = {}
        self.max_workers = max_workers
    
    def process_query(self, query: str, search_engine: str = 'google', max_results: int = 3) -> str:
        """Process a user query with fallback search and parallel processing"""
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Search with fallback
            search_results = self.searcher.search_with_fallback(query, search_engine, max_results)
            
            if not search_results:
                return "I couldn't find any relevant information for your query. Please try rephrasing your question."
            
            # Step 2: Extract content in parallel
            urls = [result.url for result in search_results]
            content_dict = self.extractor.extract_content_parallel(urls)
            
            # Step 3: Process documents with extracted content
            processed_documents = self._process_documents_with_content(search_results, content_dict)
            
            # Check if we have enough processed documents
            if not processed_documents:
                # Try with more results if we have few processed documents
                if len(search_results) < max_results * 2:
                    logger.info("Trying with more search results...")
                    more_results = self.searcher.search_with_fallback(query, search_engine, max_results * 2)
                    if more_results:
                        more_urls = [result.url for result in more_results]
                        more_content_dict = self.extractor.extract_content_parallel(more_urls)
                        processed_documents = self._process_documents_with_content(more_results, more_content_dict)
                
                if not processed_documents:
                    return "I found search results but couldn't extract meaningful content from them. Please try a different search term."
            
            # Step 4: Generate response using Ollama
            context = self._prepare_context(processed_documents)
            response = self.llm.generate_response(query, context)
            
            # Step 5: Add source attribution with engine info
            sources = []
            for doc in processed_documents[:2]:
                engine_info = f"({doc.source_engine})" if doc.source_engine != "unknown" else ""
                sources.append(f"{doc.url} {engine_info}")
            
            response += f"\n\nSources: {', '.join(sources)}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error while processing your query: {str(e)}. Please try again."
    
    def _process_documents_with_content(self, search_results: List[SearchResult], content_dict: Dict[str, Optional[str]]) -> List[ProcessedDocument]:
        """Process documents with pre-extracted content"""
        processed_docs = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_result = {
                executor.submit(self._process_single_document_with_content, result, content_dict.get(result.url)): result 
                for result in search_results
            }
            
            for future in as_completed(future_to_result):
                result = future_to_result[future]
                try:
                    processed_doc = future.result()
                    if processed_doc:
                        processed_docs.append(processed_doc)
                except Exception as e:
                    logger.error(f"Failed to process document {result.url}: {e}")
        
        return processed_docs
    
    def _process_single_document_with_content(self, search_result: SearchResult, content: Optional[str]) -> Optional[ProcessedDocument]:
        """Process a single search result document with pre-extracted content"""
        # Check cache first
        cache_key = hashlib.md5(search_result.url.encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Use pre-extracted content if available
            if not content:
                return None
            
            # Extract metadata
            metadata = self.extractor.extract_metadata(search_result.url)
            
            # Process with Ollama
            summary = self.llm.summarize_content(content)
            key_points = self.llm.extract_key_points(content)
            
            # Calculate source credibility
            credibility = self._calculate_credibility(search_result.url, metadata)
            
            processed_doc = ProcessedDocument(
                url=search_result.url,
                title=metadata.get('title', search_result.title),
                content=content,
                summary=summary,
                key_points=key_points,
                source_credibility=credibility,
                source_engine=search_result.source_engine
            )
            
            # Cache the result
            self.cache[cache_key] = processed_doc
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document {search_result.url}: {e}")
            return None
    
    def _extract_and_process_documents(self, search_results: List[SearchResult]) -> List[ProcessedDocument]:
        """Legacy method - kept for compatibility"""
        return self._process_documents_with_content(search_results, {})
    
    def _process_single_document(self, search_result: SearchResult) -> Optional[ProcessedDocument]:
        """Legacy method - kept for compatibility"""
        content = self.extractor.extract_content(search_result.url)
        return self._process_single_document_with_content(search_result, content)
    
    def _prepare_context(self, documents: List[ProcessedDocument]) -> str:
        """Prepare context from processed documents for Ollama"""
        context_parts = []
        
        for i, doc in enumerate(documents[:2], 1):  # Use top 2 documents
            engine_info = f" (via {doc.source_engine})" if doc.source_engine != "unknown" else ""
            context_parts.append(f"Source {i}: {doc.title}{engine_info}")
            context_parts.append(f"Content: {doc.summary}")
            if doc.key_points:
                context_parts.append(f"Key points: {'; '.join(doc.key_points)}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _calculate_credibility(self, url: str, metadata: Dict[str, Any]) -> float:
        """Calculate source credibility score"""
        credibility = 0.5  # Base score
        
        # Check domain
        domain = urlparse(url).netloc.lower()
        trusted_domains = [
            'wikipedia.org', 'edu', 'gov', 'org', 
            'bbc.com', 'reuters.com', 'ap.org', 'nytimes.com',
            'wsj.com', 'forbes.com', 'techcrunch.com'
        ]
        
        for trusted in trusted_domains:
            if trusted in domain:
                credibility += 0.3
                break
        
        # Check for HTTPS
        if url.startswith('https://'):
            credibility += 0.1
        
        # Check metadata completeness
        if metadata.get('description'):
            credibility += 0.1
        
        return max(0.0, min(credibility, 1.0))

def main():
    """Main function to demonstrate the RAG agent with fallback and parallel processing"""
    print("=== RAG Agent with Fallback Search & Parallel Processing ===")
    print("This agent uses Ollama for local LLM processing with robust search fallback.")
    print("Features: Fallback search engines, parallel content extraction, caching")
    print("Make sure Ollama is running: ollama serve")
    print("Available models: llama2, mistral, codellama, gemma-3n-e4b-it, etc.")
    print("Type 'quit' to exit.\n")
    
    
    model_name = "hf.co/unsloth/gemma-3n-E4B-it-GGUF:UD-Q4_K_XL"
    
    # Get parallel processing preference
    max_workers = input("Enter number of parallel workers [default: 4]: ").strip()
    if not max_workers:
        max_workers = 4
    else:
        try:
            max_workers = int(max_workers)
        except ValueError:
            max_workers = 4
    
    try:
        agent = RAGAgent(model_name=model_name, max_workers=max_workers)
        print(f"✓ Connected to Ollama with model: {model_name}")
        print(f"✓ Parallel processing enabled with {max_workers} workers")
        print(f"✓ Search engines available: {list(agent.searcher.engine_availability.keys())}")
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        print("Please make sure Ollama is running and the model is installed.")
        return
    
    while True:
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        # Ask for search engine preference
        engine = input("Preferred search engine (google/duckduckgo) [default: google]: ").strip().lower()
        if not engine or engine not in ['google', 'duckduckgo']:
            engine = 'google'
        
        print(f"\nSearching with fallback for: {query}")
        print("Processing with parallel extraction...")
        
        try:
            response = agent.process_query(query, engine)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"Error processing query: {e}\n")

if __name__ == "__main__":
    main()
