import os
import csv
import time
import uuid
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchFieldDataType
)
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Azure AI Search index
index_client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY))


fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="question", type=SearchFieldDataType.String, searchable=True),
    SearchableField(name="answer", type=SearchFieldDataType.String, searchable=True),
    SearchField(
        name="vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=3072,
        vector_search_profile_name="faq-vector-config",
    ),
]

vector_search = VectorSearch(
    profiles=[
        VectorSearchProfile(
            name="faq-vector-config",
            algorithm_configuration_name="faq-algorithms-config",
        )
    ],
    algorithms=[HnswAlgorithmConfiguration(name="faq-algorithms-config")],
)

index = SearchIndex(name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)

if index_client.get_index(AZURE_SEARCH_INDEX_NAME):
    print(
        f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists. Deleting it before creating a new one."
    )
    index_client.delete_index(AZURE_SEARCH_INDEX_NAME)

index_client.create_or_update_index(index)

# Generate embeddings for FAQ data and upload to Azure AI Search
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION
)
docs = []
faq_data_path = os.path.join("data", "faq.csv")
start_time = time.time()

with open(faq_data_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        resp = openai_client.embeddings.create(input=row["question"], model=AZURE_OPENAI_EMBEDDING_NAME)
        vector = resp.data[0].embedding
        if vector:
            docs.append(
                {
                    "id": str(uuid.uuid4()),
                    "question": row["question"],
                    "answer": row["answer"],
                    "vector": vector,
                }
            )

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.5f} seconds")

# Upload the generated documents to Azure AI Search
search_client = SearchClient(
    AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, AzureKeyCredential(AZURE_SEARCH_KEY)
)
search_client.merge_or_upload_documents(documents=docs)
print(f"Indexed {len(docs)} documents.")
