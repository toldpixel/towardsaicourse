import os
import chromadb
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleKeywordTableIndex # create a SimpleKeywordTableIndex object, which will act as the index for the keyword search.
from hybrid_retriever import HybridRetriever # custom HybridRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.evaluation import generate_question_context_pairs
import pandas as pd
from llama_index.core.evaluation import RetrieverEvaluator
import asyncio

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
"""
we need to create a keyword index that will help with the keyword search to create a hybrid retriever later
"""

def retrieve_all_nodes_from_vector_index(vector_index, query="Whatever", similarity_top_k=100000000):
    # Set similarity_top_k to a large number to retrieve all the nodes
    vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)

    # Retrieve all nodes
    all_nodes = vector_retriever.retrieve(query)
    nodes = [item.node for item in all_nodes]

    return nodes

nodes = retrieve_all_nodes_from_vector_index(vector_index)
print(len(nodes)) # 5834

# Define the KeyworddTableIndex using all the nodes.
keyword_index = SimpleKeywordTableIndex(nodes=nodes)


######################### Hybrid Retriever #########################################

# Create hybrid query engine
vector_retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=6)
keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, num_chunks_per_query=6)
hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever, max_retrieve=6)
response_synthesizer = get_response_synthesizer(llm=Settings.llm)
hybrid_query_engine = RetrieverQueryEngine(
    retriever=hybrid_retriever,
    response_synthesizer=response_synthesizer,
)

############################## Test the Hybrid retriever ##########################
answer = hybrid_query_engine.query("How does KOSMOS-2 work?")
print(answer)

############################## Test the Vector retriever ##########################
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)

# Test the query engine
answer = vector_query_engine.query("How does KOSMOS-2 work?")
print(answer)



############################## EVALUATION ##########################
# Create questions for each segment. These questions will be used to
# assess whether the retriever can accurately identify and return the
# corresponding segment when queried.
rag_eval_dataset = generate_question_context_pairs(
    nodes, llm=Settings.llm, num_questions_per_chunk=1
)

# We can save the evaluation dataset as a json file for later use.
rag_eval_dataset.save_json("./rag_eval_dataset_question_context.json")

## Load the dataset 
# rag_eval_dataset = EmbeddingQAFinetuneDataset.from_json("./rag_eval_dataset_question_context_subset_50.json")

#  A simple function to show the evaluation result.
def from_eval_results_to_dataframe(name, eval_results):
    """Convert evaluation results to a pandas dataframe."""
    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"Retriever Name": [name], "Hit Rate": [hit_rate], "MRR": [mrr]}
    )

    return metric_df

# We can evaluate the retievers with different top_k values.
for i in [2, 4, 6, 8, 10]:
    # Evaluate hybrid retriever
    vector_retriever = VectorIndexRetriever(
		    index=vector_index, similarity_top_k=i
	  )
    keyword_retriever = KeywordTableSimpleRetriever(
		    index=keyword_index, num_chunks_per_query=i
	  )
    hybrid_retriever = HybridRetriever(
		    vector_retriever, keyword_retriever, max_retrieve=i
		)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=hybrid_retriever
    )
    eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(rag_eval_dataset))
    print(from_eval_results_to_dataframe(f"Hybrid retriever top_{i}", eval_results))

    # Evaluate vector retriever
    vector_retriever = VectorIndexRetriever(
		    index=vector_index, similarity_top_k=i
		)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=vector_retriever
    )
    eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(rag_eval_dataset))
    print(from_eval_results_to_dataframe(f"Vector retriever top_{i}", eval_results))