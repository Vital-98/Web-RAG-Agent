LLM_MODEL = 'gemma3n:2b'  
LLM_TIMEOUT = 30  


MAX_SEARCH_RESULTS = 5  
SEARCH_TIMEOUT = 15  
ENABLE_DUCKDUCKGO_FALLBACK = True  


EXTRACTION_TIMEOUT = 15  
MIN_CONTENT_LENGTH = 100  
MAX_CHUNK_SIZE = 1200  
ENABLE_TRAFILATURA = True  
ENABLE_BEAUTIFULSOUP_FALLBACK = True 


USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
REQUEST_DELAY = 1  


PLAYWRIGHT_HEADLESS = True 
PLAYWRIGHT_TIMEOUT = 10000  

# Output 
SHOW_TIMING = True  
SHOW_SOURCES = True  
VERBOSE_LOGGING = False 


GOOGLE_SEARCH_URL = "https://www.google.com/search?q={query}"
DUCKDUCKGO_SEARCH_URL = "https://duckduckgo.com/html/?q={query}"

# Content Filtering
BLOCKED_DOMAINS = [
    'google.com',
    'facebook.com',
    'twitter.com',
    'instagram.com',
    'youtube.com'
]

BLOCKED_CONTENT_TYPES = [
    'script',
    'style',
    'nav',
    'header',
    'footer',
    'aside',
    'advertisement'
]


RANKING_ENABLED = True  
MAX_RANKED_CHUNKS = 8  
QUALITY_WEIGHT = 0.4 
RELEVANCE_WEIGHT = 0.6 

# Parallel scrap
PARALLEL_SCRAPING_ENABLED = True  
MAX_CONCURRENT_SCRAPES = 5  
SCRAPING_TIMEOUT_PER_URL = 15  
ENABLE_RATE_LIMITING = True  
RATE_LIMIT_DELAY = 0.5  


MAX_RETRIES = 3  
RETRY_DELAY = 2 
ENABLE_CAPTCHA_DETECTION = True  
ENABLE_RETRY_MECHANISMS = True  

# Scoring
MIN_QUALITY_SCORE = 0.1  
OPTIMAL_CONTENT_LENGTH_MIN = 500  
OPTIMAL_CONTENT_LENGTH_MAX = 2000  
MIN_SENTENCE_LENGTH = 10  
MAX_SENTENCE_LENGTH = 25  
OPTIMAL_PARAGRAPH_COUNT_MIN = 3  
OPTIMAL_PARAGRAPH_COUNT_MAX = 10  
MIN_DIVERSITY_RATIO = 0.5 

# Relevance sscoring 
TITLE_RELEVANCE_WEIGHT = 0.4  
CONTENT_RELEVANCE_WEIGHT = 0.3  
PROXIMITY_WEIGHT = 0.2  
SEMANTIC_WEIGHT = 0.1  

# Domain sources for quality scoring bonus
CREDIBLE_DOMAINS = [
    'wikipedia.org',
    'mit.edu',
    'stanford.edu',
    'harvard.edu',
    'nature.com',
    'science.org',
    'arxiv.org',
    'ieee.org',
    'acm.org',
    'springer.com'
]

# Captcha detection 
CAPTCHA_PATTERNS = [
    r'captcha',
    r'verify.*human',
    r'robot.*check',
    r'security.*check',
    r'cloudflare',
    r'ddos.*protection',
    r'rate.*limit',
    r'access.*denied',
    r'blocked.*request',
    r'bot.*detection'
]

# browser behavior
BROWSER_HEADERS = {
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache'
}

