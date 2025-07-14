# A minimal approach to RAG using Azure AI Search

🔍🤖💬 A minimal RAG (Retrieval-Augmented Generation) implementation that indexes FAQ data into Azure AI Search with vector embeddings and provides a chat terminal using Azure OpenAI for answering questions based on the retrieved content.

## Setup
1. Copy `.env.example` → `.env` and add your Azure endpoints & keys  
2. poetry install

## Ingest FAQ

- To use Vector Search in Azure AI Search, you need a vectorizer, which uses the Azure OpenAI Embedding model to turn your text into a numerical embedding. If you prefer, you can also create the embedding yourself and provide it directly to Azure AI Search for Vector Search. 
- Options 1 and 3 demonstrate how to provide the embedding directly without using a vectorizer. - `chat_app.py` uses `VectorizedQuery`. 
- Options 2 and 4 demonstrate the use of a vectorizer along with semantic search. - `chat_app_v2.py` uses `VectorizableTextQuery`.

1. 📤Push - upload local data
    ```python
    python push_aisearch_index.py 
    ```
2. 📤Push - upload local data w/ Vectorizer and Semantic Search
    ```python
    python push_aisearch_index_v2.py 
    ```
3. 📥Pull - using Azure AI Search Indexer
    ```python
    python pull_aisearch_index.py 
    ```
4. 📥Pull - using Azure AI Search Indexer w/ Vectorizer and Semantic Search
    ```python
    python pull_aisearch_index_v2.py 
    ```

## Chat

- To utilize an embedding directly with `VectorizedQuery`
    ```python
    python chat_app.py 
    ```
- To utilize text with a vectorizer, use the `VectorizableTextQuery` method.
    ```python
    python chat_app_v2.py 
    ```
- Type your question or `exit` to quit.

## Azure AI Foundry

To connect Azure AI Search with Azure AI Foundry, you need to add both a vectorizer and semantic search to the index. [Use an existing AI Search index with the Azure AI Search tool](https://review.learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/azure-ai-search?branch=main&tabs=azurecli)

📚 Learn more: [csv indexer](https://github.com/Azure/azure-search-vector-samples/tree/main/demo-python/code/indexers/csv-indexer)
