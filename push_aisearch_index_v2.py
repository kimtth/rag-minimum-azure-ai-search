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
    SearchFieldDataType,
    AzureOpenAIModelName,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticConfiguration,
    SemanticSearch,
    SemanticPrioritizedFields,
    SemanticField,
)
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_NAME")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


# Create Azure AI Search index
index_client = SearchIndexClient(
    AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY)
)

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="question", type=SearchFieldDataType.String, searchable=True, retrievable=True),
    SearchableField(name="answer", type=SearchFieldDataType.String, searchable=True, retrievable=True),
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
            vectorizer_name="faq-vectorizer",
        )
    ],
    algorithms=[HnswAlgorithmConfiguration(name="faq-algorithms-config")],
    vectorizers=[  
        AzureOpenAIVectorizer(  
            vectorizer_name="faq-vectorizer",  
            parameters=AzureOpenAIVectorizerParameters(  
                resource_url=AZURE_OPENAI_ENDPOINT,  
                deployment_name=AZURE_OPENAI_EMBEDDING_NAME,
                model_name=AzureOpenAIModelName.TEXT_EMBEDDING3_LARGE,
                api_key=AZURE_OPENAI_API_KEY,
            ),
        ),  
    ],
)

semantic_config = SemanticConfiguration(  
    name="faq-semantic-config",  
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="question"),
        content_fields=[SemanticField(field_name="answer")]  
    ),  
)

semantic_search = SemanticSearch(configurations=[semantic_config])  

index = SearchIndex(
    name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search, semantic_search=semantic_search
)

# Delete existing index if it exists before creating a new one
try:
    index_client.get_index(AZURE_SEARCH_INDEX_NAME)
    print(
        f" '{AZURE_SEARCH_INDEX_NAME}' is already exists. Deleting it before creating a new index."
    )
    index_client.delete_index(AZURE_SEARCH_INDEX_NAME)
except ResourceNotFoundError:
    pass

index_client.create_or_update_index(index)

# Call Azure OpenAI directly to generate embedding vectors.
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)
docs = []
faq_data_path = os.path.join("data", "faq.csv")
start_time = time.time()

# Read FAQ data from CSV file, generate embedding vectors, and add them to documents for indexing
with open(faq_data_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        resp = openai_client.embeddings.create(
            input=row["question"], model=AZURE_OPENAI_EMBEDDING_NAME
        )
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

# Index the documents into Azure AI Search
search_client = SearchClient(
    AZURE_SEARCH_ENDPOINT,
    AZURE_SEARCH_INDEX_NAME,
    AzureKeyCredential(AZURE_SEARCH_KEY)
)
search_client.merge_or_upload_documents(documents=docs)
print(f"Indexed {len(docs)} documents.")
