import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes import SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    SearchFieldDataType,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    FieldMapping,
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIModelName,
    SearchIndexerSkillset,
    SearchIndexer,
    IndexingParameters,
    IndexingParametersConfiguration,
    SearchIndexerDataSourceType,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    BlobIndexerParsingMode,
    # BlobIndexerDataToExtract
)
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = f'{os.getenv("AZURE_SEARCH_INDEX_NAME")}-pull'
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")
AZURE_BLOB_STORAGE_CONNECTION_STRING = os.getenv("AZURE_BLOB_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

credential = AzureKeyCredential(AZURE_SEARCH_KEY)

# Initialize Azure AI Search index
index_client = SearchIndexClient(AZURE_SEARCH_ENDPOINT, credential=credential)
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

index = SearchIndex(
    name=AZURE_SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search
)
try:
    index_client.get_index(AZURE_SEARCH_INDEX_NAME)
    print(
        f"Index '{AZURE_SEARCH_INDEX_NAME}' already exists. Deleting it before creating a new one."
    )
    # index_client.delete_index(AZURE_SEARCH_INDEX_NAME)
except ResourceNotFoundError:
    pass

index_client.create_or_update_index(index)

# # Create a data source
indexer_client = SearchIndexerClient(
    endpoint=AZURE_SEARCH_ENDPOINT, credential=credential
)
container = SearchIndexerDataContainer(name=AZURE_BLOB_CONTAINER_NAME)
data_source_connection = SearchIndexerDataSourceConnection(
    name="faq-ds",
    type=SearchIndexerDataSourceType.AZURE_BLOB,
    connection_string=AZURE_BLOB_STORAGE_CONNECTION_STRING,
    container=container,
)
data_source = indexer_client.create_or_update_data_source_connection(
    data_source_connection
)
print(f"Data source '{data_source.name}' created or updated")

# # -------------------------------------------------------------------------------

# # Create a skillset
skillset_name = "faq-ss"

embedding_skill = AzureOpenAIEmbeddingSkill(
    description="Skill to generate embeddings via Azure OpenAI",
    resource_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    deployment_name=AZURE_OPENAI_EMBEDDING_NAME,
    model_name=AzureOpenAIModelName.TEXT_EMBEDDING3_LARGE,
    dimensions=3072,
    inputs=[
        InputFieldMappingEntry(
            name="text", 
            source="/document/question"
        ),
    ],
    outputs=[
        OutputFieldMappingEntry(
            name="embedding",
            target_name="emb_vector"
        )
    ],
)

skills = [embedding_skill]
skillset = SearchIndexerSkillset(
    name=skillset_name,
    description="Skillset to chunk documents and generating embeddings",
    skills=skills
)

indexer_client.create_or_update_skillset(skillset)
print(f"{skillset.name} created")


# -------------------------------------------------------------------------------

# Create an indexer
indexer_name = "faq-idxr"

indexer_parameters_config = IndexingParametersConfiguration(
    parsing_mode=BlobIndexerParsingMode.DELIMITED_TEXT,
    first_line_contains_headers=True,
    delimited_text_delimiter=",",
    # To resolve the issue: Configuration property 'queryTimeout' is not supported for the data source of type 'azureblob'.
    query_timeout=None,  # This is not required for Azure Blob Storage
    # data_to_extract=BlobIndexerDataToExtract.CONTENT_AND_METADATA
)
                  
indexer_parameters = IndexingParameters(
    configuration=indexer_parameters_config,
)
indexer = SearchIndexer(
    name=indexer_name,
    description="Indexer to index documents and generate embeddings",
    skillset_name=skillset_name,
    target_index_name=AZURE_SEARCH_INDEX_NAME,
    data_source_name=data_source.name,
    parameters=indexer_parameters,
    field_mappings=[
        FieldMapping(
            # !important: Do not use '/document' prefix to indicate the source field
            source_field_name="question", 
            target_field_name="question"
        ),
        FieldMapping(
            source_field_name="answer", 
            target_field_name="answer"
        )
    ],
    output_field_mappings=[
        FieldMapping(
            # sourceFieldName is an invalid path: path must begin with '/document'
            # !Important: The * wildcard indicates an array. It should be used to link the output of the skill to the target field
            source_field_name="/document/emb_vector/*",
            target_field_name="vector"
        )
    ]
)

indexer_result = indexer_client.create_or_update_indexer(indexer)

print(f"Indexer '{indexer_result.name}' created or updated")

# -------------------------------------------------------------------------------
