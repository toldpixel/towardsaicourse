import os 
import chromadb

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.core.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine.transform_query_engine import TransformQueryEngine

Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")

# Load the vector store from the local storage.
db = chromadb.PersistentClient(path="./content/ai_tutor_knowledge")
chroma_collection = db.get_collection("ai_tutor_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
vector_index = VectorStoreIndex.from_vector_store(vector_store)

########################### Query Dataset ###################################
"""
Default query engine
"""
# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
query_engine = vector_index.as_query_engine()

res = query_engine.query("Write about Llama 3.1 Model, BERT and PEFT")

# We print the response
res.response

"The below response indicates that the query engine could give a partial answer to the query. "
"The query engine couldn’t give answers related to the Llama 3.1 model and BERT. "
"Therefore, we consider the query to be complex as it requires the query engine to look for references in different documents."
"Note: this is a simple example that could also be solved by using a large top k and feeding more chunks and sources into the final model."

# manually evaluate the relevancy of the retrieved contexts by printing details such as node_id, title of context, textual information, and relevancy score with respect to the query.
for src in res.source_nodes:
  print("Node ID\t", src.node_id)
  print("Title\t", src.metadata['title'])
  print("Text\t", src.text)
  print("Score\t", src.score)
  print("-_"*20)

######################### Multi-Step Query Rewriting #####################################
# Using GPT-4o-mini 
# We use the StepDecomposeQueryTransform class to generate subqueries from the original query. 
step_decompose_transform_gpt4o = StepDecomposeQueryTransform(verbose=True)

# Query Engine
query_engine_gpt4o_mini = vector_index.as_query_engine()

#Multi Step Query Engine - Use multi_step_query_engine to process the query, using Chroma as the retriever and gpt-4o-mini as the generative model.
multi_step_query_engine = MultiStepQueryEngine(
    query_engine=query_engine_gpt4o_mini,
    query_transform=step_decompose_transform_gpt4o,
    index_summary = "Used to answer the Questions about RAG, Machine Learning, Deep Learning, and Generative AI, Note: Don't repeat the Same quesion",
)

#the complex query is decomposed into simpler queries, which results in a better retrieval score.
response = multi_step_query_engine.query("Write about Llama 3.1 Model, BERT and PEFT")

for query, response in response.metadata['sub_qa']:
    print(f"**{query}**\n{response}\n")

# The response of MultiStepQueryEngine is accurate compared to the default query engine version, highlighting the importance of query variation and augmentation methods.

# Using Gemini-1.5-Flash gemini-1.5-flash as the LLM for text generation. All the APIs used are the same; we only changed the LLM to gemini-1.5-flash.

########################### Subquestion Query Engine ###########################################
"""
The SubQuestionQueryEngine works by breaking down the original query into sub-questions, 
each of which pertains to a relevant data source. 
The intermediate answers from these sub-questions are used as contextual 
information and contribute to the final answer. Each sub-question extracts specific 
information from the data source it is directed to. The responses to these sub-questions 
are combined to form a comprehensive answer
"""

query_engine = vector_index.as_query_engine()

query_engine_tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="LlamaIndex",
            description="Used to answer the Questions about RAG, Machine Learning, Deep Learning, and Generative AI. Note: Don't repeat the Same question",
        ),
    ),
]

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = sub_question_engine.query("Write about Llama 3.1 Model, BERT and PEFT")


############################### HyDE Transform ################################################
# HyDE transforms the given query into a hypothetical document and then uses it for the retrieval process instead of the query.

query_engine = vector_index.as_query_engine()
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

response = hyde_query_engine.query("Write about Llama 3.1 Model, BERT and PEFT")
response.response

#Furthermore, we can check the hypothetical document generated during the query transformation phase. The hyde instance of class HyDEQueryTransform creates the hypothetical document by passing the query to it.
query_bundle = hyde("Write about Llama 3.1 Model, BERT and PEFT")
hyde_doc = query_bundle.embedding_strs[0]
print(hyde_doc)

