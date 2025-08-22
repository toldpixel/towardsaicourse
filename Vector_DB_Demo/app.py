import os
import csv
import chromadb
from dotenv import load_dotenv
from os.path import join, dirname
from llama_index.core import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini


load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")

text = ""

# Load the file as a JSON
with open("./mini-dataset.csv", mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)

    for idx, row in enumerate(csv_reader):
        if idx == 0:
            continue
        text += row[1]

chunk_size = 512
chunks = []

# Split the long text into smaller manageable chunks of 512 characters.
for i in range(0, len(text), chunk_size):
    chunks.append(text[i : i + chunk_size])

print(len(chunks))

# Convert the chunks to Document objects so the LlamaIndex framework can process them.
documents = [Document(text=t) for t in chunks]
# create client and a new collection
# chromadb.EphemeralClient saves data in-memory.
chroma_client = chromadb.PersistentClient(path="./mini-chunked-dataset")
# Delete collection on rerun if already exists
if chroma_client.get_collection("mini-chunked-dataset"):
    chroma_client.delete_collection("mini-chunked-dataset")
chroma_collection = chroma_client.create_collection("mini-chunked-dataset")

# Define a storage context object using the created vector database.
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Build index / generate embeddings using OpenAI embedding model
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    storage_context=storage_context,
    show_progress=True,
)

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
llm = Gemini(model="models/gemini-1.5-flash", temperature=1, max_tokens=512)
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)

response = query_engine.query("How many parameters LLaMA2 model has?")
print(response)