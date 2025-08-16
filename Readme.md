For this demo task I opted to consult with microsoft autogen documentation for the memory and rag system, which ultimately the agent is based upon. Other documentations and references which  were required were that of Ollama usage docs and gemma model 3n-e2b.

The flow of the agent is based as such: 
1. User Query
2. Multi-Engine web search
3. Parallel scraping
4. Content ranking system
5. Autogen memory
6. Ollama llm local implementation
7. Output answer


Features: 
Initially the agent was based upon basic rag concept, but with autogen documentation acting as insight, rag memory was added. Then using ranking content get the best output of the queries was used updated(while not necessary it does give a outlook of other results to compare various conditions and sites). 
I had used the scraping method which implements sequentially in a loop. THis =can cause bottlenecks and is also slower, so I decided to apply parallel scraping to optimise it further.
