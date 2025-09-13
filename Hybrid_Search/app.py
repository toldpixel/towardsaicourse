import os
import chromadb
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")

Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load the vector store from the local storage.
db = chromadb.PersistentClient(path="./content/ai_tutor_knowledge")
chroma_collection = db.get_collection("ai_tutor_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create the index based on the vector store.
vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


######################### CREATE THE KEYWORD INDEX #################################
