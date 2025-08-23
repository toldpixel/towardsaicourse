import os 
from dotenv import load_dotenv
import chromadb
import csv
import hashlib
import pandas as pd
import asyncio
#from llama_index.core.evaluation import generate_question_context_pairs
from evaluation.qa_pairs import generate_question_context_pairs # modified version for request delay free gemini
from correctness_set.qa_set import qa_set
from os.path import join, dirname
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.schema import BaseNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core import Settings
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator, BatchEvalRunner
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import CorrectnessEvaluator

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")

Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small"
)

######################## CREATE VECTORSTORE ####################################
# chromadb.EphemeralClient saves data in-memory.
chroma_client = chromadb.PersistentClient(path="./mini-llama-articles")
# Delete collection on rerun if already exists
if chroma_client.get_collection("mini-llama-articles"):
    chroma_client.delete_collection("mini-llama-articles")
chroma_collection = chroma_client.create_collection("mini-llama-articles")
# Define a storage context object using the created vector database.
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

######################## LOAD THE ARTICLES ####################################
rows = []
# Load the file as a JSON
with open(join(dirname(dirname(__file__)), "mini-dataset.csv"), mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)

    for idx, row in enumerate(csv_reader):
		    # Skip header row
        if idx == 0:
            continue
        rows.append(row)

######################## CONVERT TO DOCUMENTS ####################################
# Convert the chunks to Document objects so the LlamaIndex framework can process them.
documents = [
    Document(
        text=row[1],
        metadata={"title": row[0], "url": row[2], "source_name": row[3]},
    )
    for row in rows
]
# By default, the node/chunks ids are set to random uuids. To ensure same id's per run, we manually set them.
for idx, doc in enumerate(documents):
    doc.id_ = f"doc_{idx}"

######################## TRANSFORMING ####################################
def deterministic_id_func(i: int, doc: BaseNode) -> str:
    """Deterministic ID function for the text splitter.
    This will be used to generate a unique repeatable identifier for each node."""
    unique_identifier = doc.id_ + str(i)
    hasher = hashlib.sha256()
    hasher.update(unique_identifier.encode("utf-8"))
    return hasher.hexdigest()


text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128, id_func=deterministic_id_func
)

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        OpenAIEmbedding(model = 'text-embedding-3-small'),
    ],
    vector_store=vector_store
)

nodes = pipeline.run(documents=documents, show_progress=True)

######################## LOAD INDEXES ####################################

index = VectorStoreIndex.from_vector_store(vector_store)

# Gemini-1.5-flash model
llm = Gemini(model="models/gemini-1.5-flash", temperature=1, max_tokens=512)
# Query
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
res = query_engine.query("How many parameters LLaMA 2 model has?")

for src in res.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata["title"])
    print("Text\t", src.text)
    print("Score\t", src.score)
    print("-_" * 20)

######################## EVALUATE RETRIEVER ####################################
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

if os.path.exists("./rag_eval_dataset.json"):
    rag_eval_dataset = EmbeddingQAFinetuneDataset.from_json(
        "./rag_eval_dataset.json"
    )
else:
    rag_eval_dataset = generate_question_context_pairs(
        nodes[:25],
        llm=llm,
        num_questions_per_chunk=1,
        request_delay=4
    )
    rag_eval_dataset.save_json("./rag_eval_dataset.json")

# We can save the dataset as a json file for later use.
rag_eval_dataset.save_json("./rag_eval_dataset.json")

# We can evaluate the retievers with different top_k values.
for i in [2, 4, 6, 8, 10]:
    retriever = index.as_retriever(similarity_top_k=i)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )
    eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(rag_eval_dataset, workers=1))
    print(display_results_retriever(f"Retriever top_{i}", eval_results))

######################## EVALUATE RESPONSE ####################################
# Define an LLM as a judge
llm_gpt4o = OpenAI(temperature=0, model="gpt-4o")
llm_gpt4o_mini = OpenAI(temperature=0, model="gpt-4o-mini")

# Initiate the faithfulnes and relevancy evaluator objects
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm_gpt4o)
relevancy_evaluator = RelevancyEvaluator(llm=llm_gpt4o)

# Extract the questions from the dataset
queries = list(rag_eval_dataset.queries.values())
# Limit to first 20 question to save time (!!remove this line in production!!)
batch_eval_queries = queries[:1]

# The batch evaluator runs the evaluation in batches
runner = BatchEvalRunner(
    {"faithfulness": faithfulness_evaluator, "relevancy": relevancy_evaluator},
    workers=1,
)

# Define a for-loop to try different `similarity_top_k` values
for i in [2, 4, 6, 8, 10]:
    # Set query engine with different number of returned chunks
    query_engine = index.as_query_engine(similarity_top_k=i, llm = llm_gpt4o_mini)

    # Run the evaluation
    eval_results = asyncio.run(runner.aevaluate_queries(query_engine, queries=batch_eval_queries))

    # Printing the results
    faithfulness_score = sum(
        result.passing for result in eval_results["faithfulness"]
    ) / len(eval_results["faithfulness"])
    print(f"top_{i} faithfulness_score: {faithfulness_score}")

    relevancy_score = sum(result.passing for result in eval_results["relevancy"]) / len(
        eval_results["relevancy"]
    )
    print(f"top_{i} relevancy_score: {relevancy_score}")
    print("="*15)

######################## EVALUATE CORRECTNESS ####################################
evaluator = CorrectnessEvaluator(llm=llm_gpt4o)

result = evaluator.evaluate(
    query=qa_set["query"],
    response=qa_set["response"],
    reference=qa_set["reference"],
)

print(result.score)
print(result.feedback)