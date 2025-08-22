import os
import nest_asyncio
from dotenv import load_dotenv
from os.path import join, dirname
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini


import chromadb
import csv

load_dotenv()
nest_asyncio.apply()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")



Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

# create client and a new collection
# chromadb.EphemeralClient saves data in-memory.
chroma_client = chromadb.PersistentClient(path="./mini-llama-articles")

# Delete collection on rerun if already exists
if chroma_client.get_collection("mini-llama-articles"):
    chroma_client.delete_collection("mini-llama-articles")

chroma_collection = chroma_client.create_collection("mini-llama-articles")

# Define a storage context object using the created vector database.
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

rows = []

# Load the file as a JSON
with open("./mini-dataset.csv", mode="r", encoding="utf-8") as file:
  csv_reader = csv.reader(file)

  for idx, row in enumerate(csv_reader):
    if idx == 0: continue; # Skip header row
    rows.append(row)

print(len(rows))

# Convert the chunks to Document objects so the LlamaIndex framework can process them.
"""
metadata provides additional information that enhances document retrieval systems by adding context and facilitating source tracking for query responses. 
This information is anything but the text in our content, such as filenames, URLs, or categories, which can be integrated into vector stores alongside document embeddings.
"""
documents = [Document(text=row[1], metadata={"title": row[0], "url": row[2], "source_name": row[3]}) for row in rows]

print(documents[0].metadata)

# Define the splitter object that split the text into segments with 512 tokens,
# with a 128 overlap between the segments.
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

# Create the pipeline to apply the transformation (splitting and embedding) on each chunk,
# and store the transformed text in the chroma vector store.
pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        OpenAIEmbedding(model = 'text-embedding-3-small'),
    ],
    vector_store=vector_store
)

# Run the transformation pipeline.
nodes = pipeline.run(documents=documents, show_progress=True)

# Load the vector store from the local storage.
db = chromadb.PersistentClient(path="./mini-llama-articles")
chroma_collection = db.get_or_create_collection("mini-llama-articles")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create the index based on the vector store.
"""
We can now create an index using the processed nodes in the vector store. 
This index enables querying by leveraging the embeddings, allowing us to search and retrieve relevant information from the vector store efficiently.
"""
index = VectorStoreIndex.from_vector_store(vector_store)

llm = Gemini(model="models/gemini-1.5-flash", temperature=1, max_tokens=512)

# Define a query engine that is responsible for retrieving related pieces of text, and using a LLM to formulate the final answer.
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
res = query_engine.query("How many parameters LLaMA2 model has?")

query_engine = index.as_query_engine(response_mode="refine", llm=llm)
# Or the following line if you want to use "tree_sumarize" method.
# query_engine = index.as_query_engine(response_mode="tree_summarize")
# Show the retrieved nodes
for src in res.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata["title"])
    print("Text\t", src.text)
    print("Score\t", src.score)
    print("-_" * 20)

res = query_engine.query("How many parameters LLaMA2 model has?")

print(res.response)

