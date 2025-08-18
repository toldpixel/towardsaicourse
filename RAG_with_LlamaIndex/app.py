import os
from dotenv import load_dotenv
from os.path import join, dirname
import csv
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
llm = Gemini(model="models/gemini-1.5-flash", temperature=1, max_tokens=512)

rows = []

# Load the CSV file
with open(join(dirname(dirname(__file__)), "mini-dataset.csv"), mode="r", encoding="utf-8") as file:
  csv_reader = csv.reader(file)

  for idx, row in enumerate(csv_reader):
    if idx == 0: continue; # Skip header row
    rows.append(row)

# The number of artickes in the dataset.
print("number of articles:", len(rows))

# Convert the texts to Document objects so the LlamaIndex framework can process them.
documents = [Document(text=row[1]) for row in rows]

# Build index / generate embeddings using OpenAI embedding model
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model="text-embedding-3-small"),
    transformations=[SentenceSplitter(chunk_size=768, chunk_overlap=64)],
    show_progress=True,
)

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("How many parameters LLaMA2 model has?")
print(response)