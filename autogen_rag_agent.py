import asyncio
import json
import time
import re
from typing import List, Dict, Optional, Tuple
from playwright.async_api import async_playwright, Page, Browser
import trafilatura
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import hashlib
from dataclasses import dataclass
from enum import Enum

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient


from config import *


class ContentQuality(Enum):
    """Content quality levels for ranking"""
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    POOR = 2
    VERY_POOR = 1


@dataclass
class RankedChunk:
    """Represents a ranked content chunk"""
    content: str
    title: str
    url: str
    quality_score: float
    relevance_score: float
    final_score: float
    metadata: Dict
    source_rank: int


class ScrapingError(Exception):
    """Custom exception for scraping errors"""
    def __init__(self, message: str, error_type: str, url: str):
        self.message = message
        self.error_type = error_type
        self.url = url
        super().__init__(self.message)


class AutoGenRAGAgent:
    def __init__(self, 
                 model_name: str = LLM_MODEL,
                 use_openai: bool = False,
                 openai_api_key: str = None):
        self.model_name = model_name
        self.use_openai = use_openai
        self.openai_api_key = openai_api_key
        
       system
        self.memory = ListMemory()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT
        })
        
        # Scraping retry 
        self.max_retries = 3
        self.retry_delay = 2
        self.captcha_detection_patterns = [
            r'captcha',
            r'verify.*human',
            r'robot.*check',
            r'security.*check',
            r'cloudflare',
            r'ddos.*protection'
        ]
        
        # 
        self.assistant_agent = self._create_autogen_agent()
    
    def _create_autogen_agent(self):
        """Create AutoGen assistant agent with appropriate model client"""
        if self.use_openai and self.openai_api_key:
            model_client = OpenAIChatCompletionClient(
                model="gpt-4o-2024-08-06",
                api_key=self.openai_api_key
            )
        else:
            
            model_client = OllamaChatCompletionClient(
                model=self.model_name,
                base_url="http://localhost:11434"
            )
        
        return AssistantAgent(
            name="rag_assistant",
            model_client=model_client,
            tools=[self._web_search_tool, self._extract_content_tool],
            memory=[self.memory],
            system_message="""You are an intelligent RAG (Retrieval-Augmented Generation) assistant that can search the web and provide accurate, concise answers based on retrieved information. 

Your capabilities:
1. Search the web for relevant information
2. Extract and process content from web pages
3. Provide well-structured answers with source attribution
4. Maintain context across conversations using memory
5. Use only the highest quality, most relevant content chunks

Always be helpful, accurate, and provide source links when possible."""
        )
    
    async def _web_search_tool(self, query: str, max_results: int = 5) -> str:
        """Tool for web search that can be called by AutoGen agent"""
        try:
            results = await self.search_web(query, max_results)
            if not results:
                return "No search results found for the query."
            
            # Format results for the agent
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(f"{i}. {result['title']}\n   URL: {result['url']}\n   Source: {result['source']}")
            
            # Store search results in memory for future reference
            await self.memory.add(MemoryContent(
                content=f"Web search results for '{query}': {json.dumps(results, indent=2)}",
                mime_type=MemoryMimeType.TEXT,
                metadata={"type": "search_results", "query": query, "timestamp": time.time()}
            ))
            
            return f"Found {len(results)} search results:\n\n" + "\n\n".join(formatted_results)
        
        except Exception as e:
            return f"Error during web search: {str(e)}"
    
    async def _extract_content_tool(self, url: str) -> str:
        """Tool for content extraction that can be called by AutoGen agent"""
        try:
            content = await self.scrape_and_extract_robust(url, "Web Page")
            if not content:
                return f"Failed to extract content from {url}"
            
            # Store extracted content in memory
            await self.memory.add(MemoryContent(
                content=f"Content extracted from {url}: {content['content'][:500]}...",
                mime_type=MemoryMimeType.TEXT,
                metadata={"type": "extracted_content", "url": url, "method": content['method']}
            ))
            
            return f"Successfully extracted content from {url} using {content['method']}. Content length: {len(content['content'])} characters."
        
        except Exception as e:
            return f"Error extracting content from {url}: {str(e)}"
    
    async def search_web(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict[str, str]]:
        """Enhanced web search with multiple search engines and error handling"""
        results = []
        
        
        try:
            google_results = await self._google_search_robust(query, max_results)
            results.extend(google_results)
        except Exception as e:
            print(f"Google search failed: {e}")
        
        #  trying alternative as DuckDuckGo
        if len(results) < max_results and ENABLE_DUCKDUCKGO_FALLBACK:
            try:
                ddg_results = await self._duckduckgo_search_robust(query, max_results - len(results))
                results.extend(ddg_results)
            except Exception as e:
                print(f"DuckDuckGo search failed: {e}")
        
        return results[:max_results]
    
    async def _google_search_robust(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Robust Google search with captcha detection and retry logic"""
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        results = []
        
        for attempt in range(self.max_retries):
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=PLAYWRIGHT_HEADLESS)
                    page = await browser.new_page()
                    
                    try:
                        #  browser behavior
                        await page.set_extra_http_headers({
                            'Accept-Language': 'en-US,en;q=0.9',
                            'Accept-Encoding': 'gzip, deflate, br',
                            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                        })
                        
                        # Randomize user agent slightly
                        await page.set_extra_http_headers({
                            'User-Agent': f'{USER_AGENT} (Attempt {attempt + 1})'
                        })
                        
                        await page.goto(search_url, wait_until='networkidle', timeout=PLAYWRIGHT_TIMEOUT)
                        
                        # Check for captcha or blocking
                        if await self._detect_captcha_or_blocking(page):
                            raise ScrapingError("Captcha or blocking detected", "captcha", search_url)
                        
                        # Handle cookie consent
                        try:
                            await page.click('button:has-text("Accept all")', timeout=3000)
                        except:
                            pass
                        
                        
                        await page.wait_for_selector('div[data-sokoban-container]', timeout=PLAYWRIGHT_TIMEOUT)
                        
                        # Extract search results
                        search_results = await page.query_selector_all('div[data-sokoban-container]')
                        
                        for result in search_results[:max_results]:
                            try:
                                title_elem = await result.query_selector('h3')
                                link_elem = await result.query_selector('a')
                                
                                if title_elem and link_elem:
                                    title = await title_elem.inner_text()
                                    href = await link_elem.get_attribute('href')
                                    
                                    if href and href.startswith('/url?q='):
                                        #  actual URL from google redirect
                                        actual_url = href.split('/url?q=')[1].split('&')[0]
                                        if actual_url.startswith('http') and 'google' not in actual_url:
                                            results.append({
                                                'title': title.strip(),
                                                'url': actual_url,
                                                'source': 'Google'
                                            })
                            except Exception as e:
                                continue
                        
                       
                        if results:
                            break
                            
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            print(f"Google search attempt {attempt + 1} failed: {e}")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                        else:
                            raise e
                    finally:
                        await browser.close()
                        
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Google search attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e
        
        return results
    
    async def _duckduckgo_search_robust(self, query: str, max_results: int) -> List[Dict[str, str]]:
        
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        results = []
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(search_url, timeout=SEARCH_TIMEOUT)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Check for captcha or blocking
                    if self._detect_captcha_in_html(response.text):
                        raise ScrapingError("Captcha detected in DuckDuckGo", "captcha", search_url)
                    
                    result_links = soup.find_all('a', class_='result__a')
                    
                    for link in result_links[:max_results]:
                        href = link.get('href')
                        if href and href.startswith('http'):
                            results.append({
                                'title': link.get_text().strip(),
                                'url': href,
                                'source': 'DuckDuckGo'
                            })
                    
                    # If we got results, break out of retry loop
                    if results:
                        break
                        
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise e
        
        return results
    
    async def _detect_captcha_or_blocking(self, page: Page) -> bool:
        """Detect captcha or blocking on a page"""
        try:
            # Check for common captcha indicators
            captcha_selectors = [
                'iframe[src*="captcha"]',
                'div[id*="captcha"]',
                'form[action*="captcha"]',
                'div[class*="captcha"]',
                'div[class*="cloudflare"]',
                'div[class*="ddos"]'
            ]
            
            for selector in captcha_selectors:
                if await page.query_selector(selector):
                    return True
            
            # Check page content for captcha keywords
            page_content = await page.content()
            return self._detect_captcha_in_html(page_content)
            
        except Exception:
            return False
    
    def _detect_captcha_in_html(self, html_content: str) -> bool:
        """Detect captcha in HTML content"""
        html_lower = html_content.lower()
        
        for pattern in self.captcha_detection_patterns:
            if re.search(pattern, html_lower):
                return True
        
        return False
    
    async def scrape_and_extract_robust(self, url: str, title: str) -> Optional[Dict[str, str]]:
        for attempt in range(self.max_retries):
            try:
                content = await self._scrape_single_attempt(url, title)
                if content:
                    return content
                    
            except ScrapingError as e:
                print(f"Scraping attempt {attempt + 1} failed: {e.error_type} - {e.message}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"All scraping attempts failed for {url}")
                    return None
                    
            except Exception as e:
                print(f"Unexpected error in scraping attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    return None
        
        return None
    
    async def _scrape_single_attempt(self, url: str, title: str) -> Optional[Dict[str, str]]:
        """Single attempt at content extraction"""
        try:
            response = self.session.get(url, timeout=EXTRACTION_TIMEOUT)
            if response.status_code != 200:
                return None

            # Checks for captcha
            if self._detect_captcha_in_html(response.text):
                raise ScrapingError("Captcha detected during content extraction", "captcha", url)

            # trying trafilatura first
            if ENABLE_TRAFILATURA:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
                    if text and len(text.strip()) > MIN_CONTENT_LENGTH:
                        return {
                            'content': text.strip(),
                            'title': title,
                            'url': url,
                            'method': 'trafilatura'
                        }

            # Fallback to BeautifulSoup
            if ENABLE_BEAUTIFULSOUP_FALLBACK:
                soup = BeautifulSoup(response.text, 'html.parser')                                
                for element_type in BLOCKED_CONTENT_TYPES:
                    for element in soup.find_all(element_type):
                        element.decompose()
                                
                text_elements = []
                
                # Trying to find main content area
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                else:
                    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                
                if text_elements:
                    text = '\n'.join([elem.get_text().strip() for elem in text_elements if elem.get_text().strip()])
                    if len(text) > MIN_CONTENT_LENGTH:
                        return {
                            'content': text,
                            'title': title,
                            'url': url,
                            'method': 'beautifulsoup'
                        }

        except Exception as e:
            print(f"[Error scraping {url}]: {e}")
        
        return None
    
    def rank_content_chunks(self, chunks: List[Dict[str, str]], query: str, max_chunks: int = 10) -> List[RankedChunk]:
        """Intelligently rank content chunks based on quality and relevance"""
        ranked_chunks = []
        
        for chunk in chunks:
            #  quality score
            quality_score = self._calculate_quality_score(chunk)
            
            #  relevance score
            relevance_score = self._calculate_relevance_score(chunk, query)
            
            #  final score (weighted combination)
            final_score = (quality_score * 0.4) + (relevance_score * 0.6)
            
            #  ranked chunk
            ranked_chunk = RankedChunk(
                content=chunk['content'],
                title=chunk['title'],
                url=chunk['url'],
                quality_score=quality_score,
                relevance_score=relevance_score,
                final_score=final_score,
                metadata=chunk.get('metadata', {}),
                source_rank=chunk.get('source_rank', 0)
            )
            
            ranked_chunks.append(ranked_chunk)
        
        # Sorting, higest first
        ranked_chunks.sort(key=lambda x: x.final_score, reverse=True)                
        return ranked_chunks[:max_chunks]
    
    def _calculate_quality_score(self, chunk: Dict[str, str]) -> float:
        content = chunk['content']
        score = 0.0
        
        # Length factor (optimal length gets highest score)
        content_length = len(content)
        if 500 <= content_length <= 2000:
            score += 0.3
        elif 200 <= content_length < 500 or 2000 < content_length <= 3000:
            score += 0.2
        elif content_length > 3000:
            score += 0.1
        
        # Readability factor
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 25:
            score += 0.2
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
            score += 0.15
        
        # Content structure factor
        paragraphs = content.split('\n\n')
        if 3 <= len(paragraphs) <= 10:
            score += 0.2
        
        # Information density factor
        words = content.split()
        unique_words = set(words)
        if len(words) > 0:
            diversity_ratio = len(unique_words) / len(words)
            if diversity_ratio > 0.7:
                score += 0.2
            elif diversity_ratio > 0.5:
                score += 0.1
        
        # Source credibility factor
        url = chunk['url']
        domain = urlparse(url).netloc.lower()
        credible_domains = ['wikipedia.org', 'mit.edu', 'stanford.edu', 'harvard.edu', 'nature.com', 'science.org']
        if any(credible in domain for credible in credible_domains):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_relevance_score(self, chunk: Dict[str, str], query: str) -> float:
        """Calculate content relevance score (0-1)"""
        content = chunk['content'].lower()
        title = chunk['title'].lower()
        query_terms = query.lower().split()
        
        score = 0.0
        
        # Title relevance (highest weight)
        title_matches = sum(1 for term in query_terms if term in title)
        if title_matches > 0:
            score += 0.4 * (title_matches / len(query_terms))
        
        # Content relevance
        content_matches = sum(1 for term in query_terms if term in content)
        if content_matches > 0:
            score += 0.3 * (content_matches / len(query_terms))
        
        # Query term proximity
        for i, term1 in enumerate(query_terms):
            for j, term2 in enumerate(query_terms):
                if i != j and term1 in content and term2 in content:
                    # Find positions of both terms
                    pos1 = content.find(term1)
                    pos2 = content.find(term2)
                    if pos1 != -1 and pos2 != -1:
                        distance = abs(pos1 - pos2)
                        if distance < 100:  # Close proximity
                            score += 0.2
                        elif distance < 500:  # Medium proximity
                            score += 0.1
        
        # Semantic similarity (simple keyword expansion)
        semantic_terms = {
            'ai': ['artificial intelligence', 'machine learning', 'neural networks'],
            'ml': ['machine learning', 'artificial intelligence', 'algorithms'],
            'python': ['programming', 'code', 'software', 'development']
        }
        
        for query_term, semantic_list in semantic_terms.items():
            if query_term in query.lower():
                for semantic_term in semantic_list:
                    if semantic_term in content:
                        score += 0.1
                        break
        
        return min(score, 1.0)
    
    async def run_rag_pipeline(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
        """Enhanced RAG pipeline with intelligent chunk ranking and parallel scraping"""
        print(f"\n Running RAG pipeline for: '{query}'")
        print("=" * 50)
        
        try:
            # Step 1: Web Search
            print(" Searching in process")
            search_results = await self.search_web(query, max_results)
            
            if not search_results:
                return " No search results found. Please try a different query."
            
            print("Parallel scraping:")
            extracted_contents = await self._parallel_scrape_content(search_results)
            
            if not extracted_contents:
                return "Failed to extract content from any of the search results."
            
            # Step 3: Content Chunking and Ranking
            print("Chunking plus ranking")
            all_chunks = []
            for content in extracted_contents:
                chunks = self.chunk_content(content['content'])
                for chunk in chunks:
                    all_chunks.append({
                        'content': chunk,
                        'title': content['title'],
                        'url': content['url'],
                        'source_rank': content['source_rank'],
                        'metadata': {'method': content['method']}
                    })
            
            # Rank chunks and select only the best
            ranked_chunks = self.rank_content_chunks(all_chunks, query, max_chunks=8)
            
            print(f"Selected {len(ranked_chunks)} top-quality chunks from {len(all_chunks)} total chunks")
            
            # Show top chunk scores
            for i, chunk in enumerate(ranked_chunks[:3], 1):
                print(f"  Top {i}: {chunk.title[:40]}... (Score: {chunk.final_score:.3f})")
            
            # Step 4: LLM Processing with AutoGen
            print(f"\nGenerating answer using {self.model_name}...")
            
            # Convert ranked chunks to format expected by AutoGen
            formatted_chunks = []
            for chunk in ranked_chunks:
                formatted_chunks.append({
                    'content': chunk.content,
                    'title': chunk.title,
                    'url': chunk.url,
                    'score': chunk.final_score
                })
            
            # Store ranked chunks in memory
            await self.memory.add(MemoryContent(
                content=f"Ranked content chunks for '{query}': {json.dumps(formatted_chunks, indent=2)}",
                mime_type=MemoryMimeType.TEXT,
                metadata={"type": "ranked_chunks", "query": query, "timestamp": time.time()}
            ))
            
            # Run the AutoGen agent
            stream = self.assistant_agent.run_stream(task=query)
            
            # Collect the response
            response = ""
            async for message in stream:
                if hasattr(message, 'content'):
                    if isinstance(message.content, list):
                        # Handle tool calls or complex content
                        for item in message.content:
                            if hasattr(item, 'content'):
                                response += str(item.content) + "\n"
                    else:
                        response += str(message.content) + "\n"
            
            return response.strip()
            
        except Exception as e:
            return f"Error running RAG pipeline: {str(e)}"
    
    async def _parallel_scrape_content(self, search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Extract content from multiple URLs in parallel with rate limiting"""
        extracted_contents = []
        
        # Check if parallel scraping is enabled
        if not PARALLEL_SCRAPING_ENABLED:
            print("Parallel scraping disabled, falling back to sequential extraction...")
            return await self._sequential_scrape_content(search_results)
        
        # Create scraping tasks
        scraping_tasks = []
        for i, result in enumerate(search_results):
            task = self._scrape_with_progress(result, i + 1, len(search_results))
            scraping_tasks.append(task)
        
        # Execute tasks with concurrency control
        max_concurrent = min(MAX_CONCURRENT_SCRAPES, len(scraping_tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def controlled_scrape(task):
            async with semaphore:
                # Add rate limiting if enabled
                if ENABLE_RATE_LIMITING and RATE_LIMIT_DELAY > 0:
                    await asyncio.sleep(RATE_LIMIT_DELAY)
                return await task
        
        # Run all scraping tasks with controlled concurrency
        print(f"Starting parallel extraction with max {max_concurrent} concurrent requests...")
        start_time = time.time()
        
        # Execute tasks with progress tracking
        completed_tasks = await asyncio.gather(*[controlled_scrape(task) for task in scraping_tasks], 
                                             return_exceptions=True)
        
        # Process results
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                print(f"Failed to extract {i+1}/{len(search_results)}: {result}")
            elif result:
                extracted_contents.append(result)
                print(f"Success {i+1}/{len(search_results)}: {len(result['content'])} chars")
            else:
                print(f"Failed {i+1}/{len(search_results)}: No content extracted")
        
        elapsed_time = time.time() - start_time
        print(f"   Parallel extraction completed in {elapsed_time:.1f}s")
        print(f"   Successfully extracted {len(extracted_contents)}/{len(search_results)} URLs")
        
        return extracted_contents
    
    async def _sequential_scrape_content(self, search_results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Fallback sequential content extraction (original method)"""
        extracted_contents = []
        
        for i, result in enumerate(search_results):
            print(f"  Extracting {i+1}/{len(search_results)}: {result['title'][:50]}...")
            content = await self.scrape_and_extract_robust(result['url'], result['title'])
            if content:
                content['source_rank'] = i + 1
                extracted_contents.append(content)
                print(f" Success ({len(content['content'])} chars)")
            else:
                print(f" Failed")
        
        return extracted_contents
    
    async def _scrape_with_progress(self, result: Dict[str, str], current: int, total: int) -> Optional[Dict[str, str]]:
        """Scrape a single URL with progress indication"""
        try:
            print(f" Extracting {current}/{total}: {result['title'][:50]}")
            content = await self.scrape_and_extract_robust(result['url'], result['title'])
            
            if content:
                content['source_rank'] = current
                return content
            else:
                return None
                
        except Exception as e:
            print(f" Error extracting {current}/{total}: {e}")
            return None
    
    def chunk_content(self, content: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
        """Split content into manageable chunks"""
        if len(content) <= max_chunk_size:
            return [content]
        
        chunks = []
        sentences = re.split(r'[.!?]+', content)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def add_to_memory(self, content: str, metadata: Dict = None):
        """Add information to the agent's memory"""
        await self.memory.add(MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
            metadata=metadata or {}
        ))
    
    async def query_memory(self, query: str) -> List[MemoryContent]:
        """Query the agent's memory for relevant information"""
        return await self.memory.query(query)
    
    async def clear_memory(self):
        """Clear all entries from memory"""
        await self.memory.clear()
    
    async def get_memory_summary(self) -> str:
        """Get a summary of what's stored in memory"""
        try:
            # This is a simplified approach - in practice you'd want more sophisticated memory querying
            return f"Memory contains {len(self.memory._memories)} entries"
        except:
            return "Memory status unavailable"


async def main():
    agent = AutoGenRAGAgent()
    
    while True:
        try:
            query = input("\n Enter your search query ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print(" Goodbye")
                break
            
            if not query:
                print("Invalid query.")
                continue
            
            
            answer = await agent.run_rag_pipeline(query)
            
            print("\n ANSWER:")
            print(answer)
            
            
            # Show memory status
            memory_summary = await agent.get_memory_summary()
            print(f"\n{memory_summary}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n An error occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    asyncio.run(main())
