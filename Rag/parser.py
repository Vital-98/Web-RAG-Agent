import requests
from bs4 import BeautifulSoup
import re
from utils import log
from config import MAX_TEXT_LENGTH

class ContentParser:
    @staticmethod
    def fetch_and_clean(url):
        log(f"Fetching content from: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
           for element in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
                element.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n+', '\n', text)  
            
            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH] + "..."
                
            log(f"Cleaned content: {len(text)} characters")
            return text
            
        except Exception as e:
            log(f"Failed to fetch {url}: {str(e)}", "ERROR")
            return ""
