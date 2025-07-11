# A minimal approach to RAG using Azure AI Search

🔍🤖💬 A minimal RAG (Retrieval-Augmented Generation) implementation that indexes FAQ data into Azure AI Search with vector embeddings and provides a chat terminal using Azure OpenAI for answering questions based on the retrieved content.

## Setup
1. Copy `.env.example` → `.env` and add your Azure endpoints & keys  
2. poetry install

## Ingest FAQ

1. Push - upload local data
```python
python push_aisearch_index.py 
```
2. Pull - using Azure AI Search Indexer
```python
python pull_aisearch_index.py 
```

## Chat
```python
python chat_app.py
```
Type your question or `exit` to quit.

📚 Learn more: [csv indexer](https://github.com/Azure/azure-search-vector-samples/tree/main/demo-python/code/indexers/csv-indexer)
