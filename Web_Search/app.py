import os
import json
import newspaper
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

# Set the "OPENAI_API_KEY" in the Python environment. Will be used by OpenAI client later.
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
GOOGLE_SEARCH_KEY = "GOOGLE_SEARCH_KEY"
GOOGLE_SEARCH_ENGINE = "GOOGLE_SEARCH_ENGINE" # Search Engine ID 

Settings.llm = OpenAI(model="gpt-4o-mini", temperature= 0)

# Embedding Model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# define sample Tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

# initialize ReAct agent
agent = ReActAgent.from_tools([multiply_tool], verbose=True) # LlM - GPT-4o-mini

res = agent.chat("What is the multiplication of 43 and 45?")

tool_spec = GoogleSearchToolSpec(key=GOOGLE_SEARCH_KEY, engine=GOOGLE_SEARCH_ENGINE)

# Wrap the google search tool to create an index on top of the returned Google search
wrapped_tool = LoadAndSearchToolSpec.from_defaults(
    tool_spec.to_tool_list()[0],
).to_tool_list()

agent = OpenAIAgent.from_tools(wrapped_tool, verbose=False)

search_results = tool_spec.google_search("LLaMA 3.2 model details")
search_results = json.loads(search_results[0].text)

pages_content = []

for item in search_results['items']:
    try:
        article = newspaper.Article( item['link'] )
        article.download()
        article.parse()
        if len(article.text) > 0:
            pages_content.append({ "url": item['link'], "text": article.text, "title": item['title'] })
    except:
        continue

print(len(pages_content))

################################## RAG Pipeline ########################################################################

# Convert the texts to Document objects so the LlamaIndex framework can process them.
documents = [Document(text=row["text"], metadata={"title": row["title"], "url": row["url"]}) for row in pages_content]

# Build index / generate embeddings using OpenAI.
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
)

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
query_engine = index.as_query_engine()

response = query_engine.query("How many parameters LLaMA 3.2 model has? list exact sizes of the models")
print(response)

# Show the retrieved nodes
for src in response.source_nodes:
  print("Title\t", src.metadata['title'])
  print("Source\t", src.metadata['url'])
  print("Score\t", src.score)
  print("-_"*20)

