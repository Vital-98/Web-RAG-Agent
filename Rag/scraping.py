from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
import time
from utils import log

class GoogleSearcher:
    def __init__(self, headless=True):
        self.options = Options()
        if headless:
            self.options.add_argument("--headless")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        self.options.page_load_strategy = 'eager'
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.implicitly_wait(10)

    def search(self, query, num_results=10):
        """Search Google and return results (title, url, snippet)."""
        log(f"🔍 Searching Google for: '{query}'")
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={num_results}"
        try:
            self.driver.get(search_url)
        except TimeoutException:
            log("Page load timed out, but continuing...", "WARNING")

        results = []
        try:
            # Wait for search results to load - multiple possible selectors
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g, .tF2Cxc, .MjjYud"))
            )
            time.sleep(2)  # Additional buffer time

            # Try multiple possible selectors for Google results
            selectors = ["div.g", ".tF2Cxc", ".MjjYud"]
            search_items = []
            
            for selector in selectors:
                search_items = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if search_items:
                    break

            for item in search_items:
                try:
                    # Try multiple title selectors
                    title_selectors = ["h3", ".LC20lb", ".DKV0Md"]
                    title = ""
                    for selector in title_selectors:
                        try:
                            title = item.find_element(By.CSS_SELECTOR, selector).text
                            if title:
                                break
                        except NoSuchElementException:
                            continue
                    
                    # Get URL
                    url = item.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                    
                    # Try multiple snippet selectors
                    snippet_selectors = [".VwiC3b", ".MUxGbd", ".lyLwlc"]
                    snippet = ""
                    for selector in snippet_selectors:
                        try:
                            snippet = item.find_element(By.CSS_SELECTOR, selector).text
                            if snippet:
                                break
                        except NoSuchElementException:
                            continue
                    
                    if title and url:  # Only add if we have basic info
                        results.append({
                            "title": title, 
                            "url": url, 
                            "snippet": snippet
                        })
                    
                    if len(results) >= num_results:
                        break
                        
                except NoSuchElementException:
                    continue

        except TimeoutException:
            log("Timeout: Search results did not load.", "ERROR")
        
        log(f"Found {len(results)} search results")
        return results

    def close(self):
        """Close the browser."""
        self.driver.quit()