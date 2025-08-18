import os
import csv
import chromadb
from dotenv import load_dotenv
from os.path import join, dirname
from langchain.schema.document import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.gemini import Gemini
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA



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
documents = [Document(page_content=t) for t in chunks]
# create client and a new collection
# chromadb.EphemeralClient saves data in-memory.
chroma_client = chromadb.PersistentClient(path="./mini-chunked-dataset")
if chroma_client.get_collection("mini-chunked-dataset"):
    chroma_client.delete_collection("mini-chunked-dataset")
chroma_collection = chroma_client.create_collection("mini-chunked-dataset")

# Define a storage context object using the created vector database.
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Add the documents to chroma DB and create Index / embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
chroma_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./mini-chunked-dataset",
    collection_name="mini-chunked-dataset",
)

# Initializing the LLM model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=512,
)

query = "How many parameters LLaMA2 model has?"
retriever = chroma_db.as_retriever(search_kwargs={"k": 4})

# Define a RetrievalQA chain that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

response = chain.invoke(query)
print(response["result"])