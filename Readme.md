In this attempt I have ensured that a strict RAG pipeline is followed and implemented.
The first stage for the project; the web scraping for information from the user query was the most crucial part as I went through many iterations and blockades in web scraping effectively. At the end I decided to use selenium-base for the scraping as it was effectively able to fetch certain number of url and results. For parsing and cleaning I used beatiful soup.

The chunking and embedding part was accomplished using sentence-transformer and its in-library emmbedder. 

While researching for RAG, cosine similarity often turned up for the determining of top chunks to be the efficent method for ranking. I did not use re-ranking even though it can be done using cross encoder  to get results on the quicker side and not to overcomplicate for this particular task.

I used the suggested gemma model from hugging face to be used for running the llm locally. My laptop has 4gb VRAM which is half of the minimumm 8GB VRAM required to efficiently run it, hnece the model processed it 100% on CPU and it takes upto 5-10 min to process the queries.

Finally the output from the rag is able to cite from sources albeit of some sites blocking access while scraping. It provides a decent summary and answer with proper citation of the ranked chunks from the best sources.
