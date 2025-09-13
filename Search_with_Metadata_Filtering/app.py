import os
import qdrant_client
import json
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.extractors import (
    KeywordExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    FilterCondition
)


"""
We opted for Qdrant because it allows us to fully utilize the filtering technique’s capabilities, 
unlike the Chroma vector store, which lacks some features (at the time of writing). 
This will also demonstrate how straightforward replacing and integrating different vector store services within the LlamaIndex framework is.
"""

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")

Settings.llm = OpenAI(temperature=0.9, model="gpt-4o-mini", max_tokens=512)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# qdrant_client = QdrantClient(location=":memory:")
# or Persist storage
qd_client = qdrant_client.QdrantClient(path="./content/")

vector_store = QdrantVectorStore(
		client=qd_client, 
		collection_name="ai_tutor_knowledge_qdrant"
)

with open("./content/ai_tutor_knowledge.jsonl", "r") as file:
    ai_tutor_knowledge = [json.loads(line) for line in file]

documents = ai_tutor_knowledge[:100]+ai_tutor_knowledge[500:]

############################## Manually generate Metadata from Keywords ##############################

# Setting the metadata fields 
def create_docs_from_list(data_list: List[dict]) -> List[Document]:
    documents = []
    for data in data_list:
        documents.append(
            Document(
                doc_id=data["doc_id"],
                text=data["content"],
                metadata={  # type: ignore
                    "url": data["url"],
                    "title": data["name"],
                    "tokens": data["tokens"],
                    "source": data["source"],
                },
                excluded_llm_metadata_keys=[
                    "title",
                    "tokens",
                    "source",
                ],
                excluded_embed_metadata_keys=[
                    "url",
                    "tokens",
                    "source",
                ],
            )
        )
    return documents

doc = create_docs_from_list(documents)
print(doc[2].metadata)

########################## Automatic Keyword Extraction ############################
"""
Luckily, LlamaIndex offers a transformation that automates the tagging process. 
This class uses an LLM to extract multiple keywords from the text. 
These keywords can normally be leveraged to enhance the context for the retriever or filter 
the chunks based on them in more advanced scenarios. This simplified technique significantly 
enhances efficiency in handling large and diverse datasets.
"""

# Define the splitter object that split the text into segments with 512 tokens,
# with a 128 overlap between the segments.
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

# Create the pipeline to apply the transformation on each chunk,
# and store the transformed text in the vector store.
pipeline = IngestionPipeline(
    transformations=[
        text_splitter, # Split documents to chunks
        KeywordExtractor(keywords=10, llm=Settings.llm), # Extract keywords
        OpenAIEmbedding(), # Convert to embedding vector
    ],
    vector_store=vector_store
)

# Run the transformation pipeline.
nodes = pipeline.run(documents=doc, show_progress=True)
print( len( nodes ) )
print(nodes[0].metadata)


##################################### Filter Based on Metadata (Keyword) ##############################################

"""
Typically, at this stage of implementing the chatbot pipeline, 
we would use the initialized vector store containing processed chunks to create our query engine for asking questions. 
However, with the advanced filtering technique, we can apply a filter to the query engine to query the knowledge base 
based on specific criteria from the filtered samples. This capability can enhance the precision and relevance of responses provided by the chatbot.

Qdrant vector store to maximize the effectiveness of this technique. Qdrant supports all listed operators, unlike Chroma, which misses a few integrations.
"""

# Create the index based on the vector store.
index = VectorStoreIndex.from_vector_store(vector_store)

MetadataFilter(key="year", operator=FilterOperator.EQ, value="2024")

filters = MetadataFilters(
    filters=[
        MetadataFilter(key="excerpt_keywords", operator=FilterOperator.TEXT_MATCH, value="PEFT"),
    ]
)

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
query_engine = index.as_query_engine(filters=filters)

res = query_engine.query("How Parameter efficient fine tuning (PEFT) Works?")

print(res.response)

# Show the retrieved nodes
for src in res.source_nodes:
  print("Node ID\t", src.node_id)
  print("Title\t", src.metadata['title'])
  print("Text\t", src.text)
  print("Score\t", src.score)
  print("Keywords\t", src.metadata['excerpt_keywords'])
  print("-_"*20)


################################################################ Filter Based on Metadata (Keyword and Source) ####################################################

"""
The example mentioned in the previous section is a basic usage of the technique with just one filter. 
It is possible to mix any number of filters, 
but keep in mind that we need to use the condition argument to specify the relationship between the individual filters. 
"""

filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="excerpt_keywords",
            operator=FilterOperator.TEXT_MATCH,
            value="BERT",
        ),
        MetadataFilter(
            key="source", operator=FilterOperator.EQ, value="tai_blog"
        ),
    ],
    condition=FilterCondition.AND,
)

# Print the output
query_engine = index.as_query_engine(filters=filters)
result = query_engine.query("Explain BERT?")

print( result.response )

# When Mismatch between Filter value and Query
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="excerpt_keywords",
            operator=FilterOperator.TEXT_MATCH,
            value="BERT",
        ),
        MetadataFilter(
            key="source", operator=FilterOperator.EQ, value="tai_blog"
        ),
    ],
    condition=FilterCondition.AND,
)

query_engine = index.as_query_engine(filters=filters)
result = query_engine.query("Explain PEFT?")
print(result.response)

# Show the retrieved nodes
for src in result.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata["title"])
    print("Text\t", src.text)
    print("Score\t", src.score)
    print("Score\t", src.metadata["excerpt_keywords"])
    print("-_" * 20)