from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from scraping_pipeline import documents

llm = OpenAI(model="gpt-4o-mini",temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)

"""
So, when using the VectorStoreIndex class to create the index, 
there is no need to pass the chunking object or specify the LLM 
when initiating a query engine using the .as_query_engine() method. 
The framework automatically picks up the necessary details from the global settings variable
"""

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

res = query_engine.query("What is a query engine?")

print(res.response)

# Show the retrieved nodes
for src in res.source_nodes:
  print("Node ID\t", src.node_id)
  print("Title\t", src.metadata['title'])
  print("URL\t", src.metadata['url'])
  print("Score\t", src.score)
  print("-_"*20)