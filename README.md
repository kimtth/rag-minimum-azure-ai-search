# 🔍🤖💬 A minimal approach to RAG using Azure AI Search

## Setup
1. Copy `.env.example` → `.env` and add your Azure endpoints & keys  
2. pip install -r requirements.txt

## Ingest FAQ
```python
python push_aisearch_index.py
```

## Chat
```python
python chat_app.py
```
Type your question or `exit` to quit.

📚 Learn more: https://learn.microsoft.com/en-us/azure/search/samples-python