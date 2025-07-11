import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
# AZURE_SEARCH_INDEX_NAME = f'{os.getenv("AZURE_SEARCH_INDEX_NAME")}-pull'
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_NAME")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")

openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Retrieve context from Azure AI Search using vector search
def retrieve_context(question, top_k=1):
    emb_resp = openai_client.embeddings.create(input=question, model=AZURE_OPENAI_EMBEDDING_NAME)
    vector_emb = emb_resp.data[0].embedding
    if vector_emb:
        try:
            vector_query = VectorizedQuery(
                vector=vector_emb,
                k_nearest_neighbors=5,
                fields="vector",
                kind="vector",
                exhaustive=True
            )

            results = search_client.search(
                # search_text=question, # Optional: When using hybrid search, this field need to be filled with a value
                vector_queries=[vector_query],
                select=["question", "answer"],
                top=5,
                include_total_count=True
            )

            print(f"Total results: {results.get_count()}")
            context = ''
            for doc in results:
                context += f"- Question: {doc['question']}, Answer: {doc['answer']}\n"
            return context.split('\n') if context else []
        except Exception as ex:
            print("Vector search failed:", ex)
    else:
        print("No vector loaded, skipping search.")


# Function to handle chat interaction with the AI assistant
def chat(question):
    context = retrieve_context(question)
    # print("Context retrieved:", context)
    system_prompt = (
        "You are an AI assistant. Use the following context to answer:\n"
        + (f"\n---\n".join(context) if context else "")
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    resp = openai_client.chat.completions.create(model=AZURE_OPENAI_DEPLOYMENT_NAME, messages=messages)
    return resp.choices[0].message.content

# Main loop for user interaction
if __name__ == "__main__":
    while True:
        q = input("You: ")
        if q.lower() in ("exit", "quit"):
            break
        ans = chat(q)
        print("AI:", ans)
