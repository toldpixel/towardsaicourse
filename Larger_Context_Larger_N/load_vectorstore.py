import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from huggingface_hub import hf_hub_download
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

vectorstore = hf_hub_download(repo_id="jaiganesan/ai_tutor_knowledge", filename="vectorstore.zip", repo_type="dataset", local_dir="./Larger_Context_Larger_N/content")

# Load the vector store from the local storage.
db = chromadb.PersistentClient(path="./Larger_Context_Larger_N/content/ai_tutor_knowledge")
chroma_collection = db.get_or_create_collection("ai_tutor_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create the index based on the vector store.
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

for i in [2, 4, 6, 8, 10, 15, 20, 25, 30]:
    query_engine = index.as_query_engine(similarity_top_k=i)

    res = query_engine.query("Explain how RAG works?")

    print(f"top_{i} results:")
    print("\t", res.response)
    print("-_" * 20)