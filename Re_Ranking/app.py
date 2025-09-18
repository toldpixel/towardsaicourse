import os
import chromadb
import asyncio
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.postprocessor.cohere_rerank import CohereRerank
import pandas as pd
from llama_index.core.evaluation import RetrieverEvaluator

Settings.llm = OpenAI(temperature=0, model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
os.environ["CO_API_KEY"] = os.environ.get("CO_API_KEY")
cohere_key = os.environ["CO_API_KEY"]


############################ Load the data ############################
# Create the vector index
db = chromadb.PersistentClient(path="./content/ai_tutor_knowledge")
chroma_collection = db.get_collection("ai_tutor_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

########################### Reranking ###################################
# Define the Cohere Reranking object to return only the first two highest ranking chunks.
# When querying the system with the question “Write about Retrieval Augmented Generation?”, the query_engine processes the question, applies reranking to retrieve relevant text chunks, and produces a response accessed via res.response.
cohere_rerank3 = CohereRerank(top_n=2, api_key=cohere_key,model = 'rerank-english-v3.0')

# Define the ServiceCotext object to tie the LLM for generating final answer,
# and the embedding model to help with retrieving related nodes.
# The `node_postprocessors` function will be applied to the retrieved nodes.
query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[cohere_rerank3]
)

res = query_engine.query("Write about Retrieval Augmented Generation?")
res.response

# Show the retrieved nodes
for src in res.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata["title"])
    print("Text\t", src.text)
    print("Score\t", src.score)
    print("-_" * 20)

########################### Evaluate #####################################
# metrics such as hit rate and Mean Reciprocal Rank (MRR) for different retrievers
#  A simple function to show the evaluation result.
def display_results_retriever(name, eval_results):
    """Display results from evaluate."""

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


# Download the evaluation dataset
#hf_hub_download(repo_id="jaiganesan/ai_tutor_knowledge", filename="rag_eval_dataset_question_context_subset_50.json", repo_type="dataset", local_dir="./")
#rag_eval_dataset = EmbeddingQAFinetuneDataset.from_json("./rag_eval_dataset_question_context_subset_50.json")


# We can evaluate the retievers with different top_k values.
for i in [2, 4, 6, 8, 10]:
    retriever= index.as_retriever(
        similarity_top_k=i, node_postprocessors=[cohere_rerank3]
    )
    retriever_evaluator= RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )
    eval_results= asyncio.run(retriever_evaluator.aevaluate_dataset(rag_eval_dataset))
    print(display_results_retriever(f"Retriever top_{i}", eval_results))


