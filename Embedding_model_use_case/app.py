import os
import asyncio
import csv
import chromadb
from dotenv import load_dotenv
from os.path import join, dirname
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)
from llama_index.core import VectorStoreIndex


"""
We can directly use HuggingFaceEmbedding to load and apply a model like intfloat/e5-small-v2
for embedding generation, providing a seamless way to switch between embedding models
without being locked into a specific provider.
"""

from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core.ingestion import IngestionPipeline

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
os.environ["CO_API_KEY"] = os.environ.get("CO_API_KEY")

Settings.llm = OpenAI(temperature=1, model="gpt-4o-mini")
llm_gpt_4o = OpenAI(temperature=0, model="gpt-4o-mini")

######################## LOAD THE ARTICLES ####################################
rows = []
# Load the file as a JSON
with open(join(dirname(dirname(__file__)), "mini-dataset.csv"), mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)

    for idx, row in enumerate(csv_reader):
        if idx == 0: continue  # Skip header row
        rows.append(row)
################################################################################

# Convert the chunks to Document objects so the LlamaIndex framework can process them.
documents = [Document(text=row[1], metadata={"title": row[0], "url": row[2], "source_name": row[3]}) for row in rows]

######################## CREATE VECTORSTORAGE OBJECT (Cohere embedding model) ####################################
# create vector store
vector_store_name = "mini-llama-articles"
chroma_client = chromadb.PersistentClient(path=vector_store_name)
# Delete collection on rerun if already exists
if chroma_client.get_collection("mini-llama-articles"):
    chroma_client.delete_collection("mini-llama-articles")
chroma_collection = chroma_client.get_or_create_collection(vector_store_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
################################################################################

######################## CREATE VECTORSTORAGE OBJECT (Opensource embedding model) ####################################
# create vector store
vector_store_name_ose = "mini-llama-articles-open-source-embed"
chroma_client_ose = chromadb.PersistentClient(path=vector_store_name)
# Delete collection on rerun if already exists
if chroma_client_ose.get_collection("mini-llama-articles"):
    chroma_client_ose.delete_collection("mini-llama-articles")
chroma_collection_ose = chroma_client_ose.get_or_create_collection(vector_store_name)
vector_store_ose = ChromaVectorStore(chroma_collection=chroma_collection_ose)
################################################################################

# Define the splitter object that splits the text into segments with 512 tokens,
# with a 128 overlap between the segments.
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)


# Create the pipeline to apply the transformation on each chunk,
# and store the transformed text in the chroma vector store.
"""
Instead of using Cohere embeddings, we can switch to an open-source model for 
creating vector embeddings. In this case, 
we’re using the intfloat/e5-small-v2 model from Hugging Face, 
one of the top-performing models for embeddings. 
"""
pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        QuestionsAnsweredExtractor(questions=3, llm=llm_gpt_4o),
        SummaryExtractor(summaries=["prev", "self"], llm=llm_gpt_4o),
        KeywordExtractor(keywords=10, llm=llm_gpt_4o),
        HuggingFaceEmbedding(model_name="intfloat/e5-small-v2"), #CohereEmbedding(model_name="embed-english-v3.0", input_type="search_document", cohere_api_key=os.environ.get("OPENAI_API_KEY1")),
    ],
    #vector_store=vector_store,
    vector_store=vector_store_ose,
)

# Run the transformation pipeline.
nodes = pipeline.run(documents=documents, show_progress=True)

# Define the Cohere Embedding Model
#embed_model = CohereEmbedding(
#		model_name="embed-english-v3.0",
#		input_type="search_query",
#		cohere_api_key=os.environ.get("CO_API_KEY")
#)
# Define the Cohere Embedding Model
embed_model_e5 = HuggingFaceEmbedding(model_name="intfloat/e5-small-v2")

# Create the index based on the vector store.
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model_e5)

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM (We set the default llm as gpt-4o-mini) to formulate the final answer.
query_engine = index.as_query_engine(similarity_top_k=5)

res = query_engine.query("How many parameters LLaMA2 model has?")

# Show the retrieved nodes
for src in res.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata["title"])
    print("Text\t", src.text)
    print("Score\t", src.score)
    print("-_" * 20)