The RAG Web Search Agent works in four main steps:

1.User Query – User enters a natural language question.

2.Web Search – The agent searches the web (Google/DuckDuckGo) for relevant pages.

3.Content Extraction – Web pages are scraped and cleaned for useful text.

4.Answer Generation – The local LLM (Gemma-3n via Ollama) processes the query + extracted content to produce a concise, accurate answer with sources.


Features: 
Web Search Integration – Finds and retrieves the most relevant online sources for a query.
Content Extraction – Uses Playwright + BeautifulSoup + Trafilatura to scrape and clean web pages into usable text.
Local LLM (Gemma-3n via Ollama) – Runs completely on your PC, avoiding APIs and vendor lock-in.
Retrieval-Augmented Generation (RAG) – Combines search results with LLM reasoning to produce context-aware answers.
Memory Support – Maintains conversational context using ListMemory for multi-turn queries.
Modular Design – Agents, search, and LLM clients are separated for easy extension or swapping components.
The model used here is gemma-3n-e2b-it as mentioned in the task : https://huggingface.co/unsloth/gemma-3n-E2B-it-GGUF/blob/main/gemma-3n-E2B-it-Q4_0.gguf

While I do have 4GB of VRAM in my GPU, it was noticed that it takes upto 5 min for queries to be processed while running locally.
The answers are on par with giving credible sources and overviews of the given queries, but can struggle due to some websites blcoking web scrapers. This can be potentially optimized and queries can be more concise in nature if autogen framework can be implemented, although it being more complex and potential API usage for best capabilities will have to addressed.

Regarding Approach:
The initial approach of autogen was diverted beacuse of many inconsistencies in dependencies and libraries used. AG2 does give more robust agent management but as it is a wrapper for autogen the latter issue was persistent. While this does change the direction of the rag implemented a bit but still the core concept of RAG remains same, infact I used AG2 RAG docs for reference and coding of the agents.


Ollama setup:
Requires ollama installed on the machine.
Model is installed using "ollama pull"
Model is started using "ollama serve"
