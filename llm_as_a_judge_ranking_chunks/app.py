import os
import chromadb
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel
from llama_index.core.prompts import PromptTemplate
from typing import (
    List,
    Optional
)
from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")

Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load the vector store from the local storage.
chroma_client = chromadb.PersistentClient(path="./content/ai_tutor_knowledge")
chroma_collection = chroma_client.create_collection("ai_tutor_knowledge")

# Re-create the vector store from the checkpoint
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create the index based on the vector store.
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

"""
The first subsection demonstrates using a pre-defined post-processor from LlamaIndex called “RankGPT,” which employs ChatGPT as a reranker.
In this case, we will use the RankGPTRerank class, 
which requires two arguments: the number of chunks to return after the reranking process and an 
LLM model tasked with comparing them in relation to a specific query
"""

rankGPT = RankGPTRerank(top_n=3, llm=Settings.llm)

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
# The `node_postprocessors` function will be applied to the retrieved nodes.
query_engine = index.as_query_engine(similarity_top_k=10, 
												node_postprocessors=[rankGPT], 
												)

res = query_engine.query("Explain how Retrieval Augmented Generation (RAG) works?")

print(res.response)
print("-_"*40)

# Show the retrieved nodes
for src in res.source_nodes:
  print("Node ID\t", src.node_id)
  print("Title\t", src.metadata['title'])
  print("Text\t", src.text)
  print("Score\t", src.score)
  print("-_"*20)


"""
The second subsection presents how to create our own custom post-processor using GPT-4o-mini to accomplish the same task.
should not be directly applied “as-is” in real-world pipelines.
Because the custom-built ranker lacks robust safeguards against malformed or incomplete model outputs and has minimal error handling for varied real-world data inputs.
"""

def judger(nodes, query):

  # The model's output template
  class OrderedNodes(BaseModel):
    """A node with the id and assigned score."""
    node_id: list
    score: list

  # Prepare the nodes and wrap them in <NODE></NODE> identifier, as well as the query
  the_nodes=""
  for idx, item in enumerate(nodes):
    the_nodes += f"<NODE{idx+1}>\nNode ID: {item.node_id}\nText: {item.text}\n</NODE{idx+1}>\n"

  query = "<QUERY>\n{}\n</QUERY>".format(query)

  # Define the prompt template
  prompt_tmpl = PromptTemplate(
    """
    You receive a qurey along with a list of nodes' text and their ids. Your task is to assign score
    to each node based on its contextually closeness to the given query. The final output is each
    node id along with its proximity score.
    Here is the list of nodes:
    {nodes_list}

    And the following is the query:
    {user_query}

    Score each of the nodes based on their text and their relevancy to the provided query.
    The score must be a decimal number between 0 an 1 so we can rank them."""
  )

  # Define the an instance of GPT-4o-mini and send the request
  llm = OpenAI(model="gpt-4o-mini")
  ordered_nodes = llm.structured_predict(
    OrderedNodes, prompt_tmpl, nodes_list=the_nodes, user_query=query
  )

  return ordered_nodes


class OpenaiAsJudgePostprocessor(BaseNodePostprocessor):
    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:

        r = judger(nodes, query_bundle)

        node_ids = r.node_id
        scores = r.score

				# Sort the scores and select the top 3
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        num_nodes_to_select = min(3, len(sorted_scores))
        top_nodes = [sorted_scores[i][0] for i in range(num_nodes_to_select)]
        # top_nodes = [sorted_scores[i][0] for i in range(3)]

				# Retrieve the ID of the selected nodes
        selected_nodes_id = [node_ids[item] for item in top_nodes]

				# Filter the nodes based on retrieved IDs
        final_nodes = []
        for item in nodes:
          if item.node_id in selected_nodes_id:
            final_nodes.append( item )

        return final_nodes
    
judge = OpenaiAsJudgePostprocessor()

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
# The `node_postprocessors` function will be applied to the retrieved nodes.
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[judge]
)

res = query_engine.query("Compare Retrieval Augmented Generation (RAG) and Parameter efficient Finetuning (PEFT)")

print( res.response )
print("-_"*40)

# Show the retrieved nodes
for src in res.source_nodes:
  print("Node ID\t", src.node_id)
  print("Title\t", src.metadata['title'])
  print("Text\t", src.text)
  print("Score\t", src.score)
  print("-_"*20)