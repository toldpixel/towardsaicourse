import os
from dotenv import load_dotenv
from llama_index.readers.web import FireCrawlWebReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

# To increase chunk size globally
# Settings.chunk_size = 2048  # or even larger like 4096
# Settings.chunk_overlap = 200

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")

# node parser with larger chunk size only for this index
node_parser = SentenceSplitter(
    chunk_size=2048,
    chunk_overlap=200,
)

firecrawl_reader = FireCrawlWebReader(
    api_key=FIRECRAWL_API_KEY,
    mode="scrape",
)

documents = firecrawl_reader.load_data(url="https://towardsai.net/")

index = VectorStoreIndex.from_documents(documents, transformations=[node_parser])
query_engine = index.as_query_engine()

res = query_engine.query("What is towards AI aim?")

print(res.response)

print("-----------------")
# Show the retrieved nodes
for src in res.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata['title'])
    print("URL\t", src.metadata['sourceURL'])
    print("Score\t", src.score)
    print("Description\t", src.metadata.get("description"))
    print("-_" * 20)