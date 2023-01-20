# Mr.Si
Bot that helps you synthesize search results using GPT-3. A smarter search.
It gets your initial question e.g. "When will we cure cancer?", then creates multiple search proposals to use on Google Search, and fetches the top urls for each of the search results.  Then condenses candidate links into answers, saving you the time of browsing each page looking for the exact answer to your question.
You can find details and examples in the SearcherC.ipynb notebook.
Mr.Si is in reference to Mr.C, an experimental Pubmed search bot built two years ago as a fun project.

# Run notebook
You can view the results of the notebook above, but to run it you will need to have api keys for openai and rapidapi which you can obtain in there respective webpages. You should create a directory called "creds". In it three text files called
- creds.txt
- rapid_api_key.txt
- rapid_api_host.txt

And paste the corresponding information in each file. Then just run the notebook.  Beware that if you increase the n of urls and n of snippets, it might take more than just a few minutes to run the final function.

You can also run the Run Searcher notebook to load all functions as a package and run queries in a clean notebook.

With 15 urls and 25 snippets, you should expect 1-3 minutes execution time and around 20 cents of OpenAI cost for the embeddings and GPT-3 completion. Unforutnately had to sleep(1) due to rate limit of rapid api, making it this slow.
GPT3 API somwtimes returns 'overloaded', but should still run fine.

# Next
If this becomes useful, next step will be to package it properly and add functionalities.  Maybe leverage LangChain for this. 
