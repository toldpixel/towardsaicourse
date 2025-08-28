import os 
from dotenv import load_dotenv
import asyncio
import chromadb
import csv
from os.path import join, dirname
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Document
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.together import TogetherLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import generate_question_context_pairs # Needed for Evaluation
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import RetrieverEvaluator, RelevancyEvaluator, FaithfulnessEvaluator, BatchEvalRunner
from llama_index.llms.openai import OpenAI
import time
from llama_index.llms.gemini import Gemini


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
os.environ["TOGETHER_AI_API_TOKEN"] = os.environ.get("TOGETHER_AI_API_TOKEN")

######################## CREATE VECTORSTORE ####################################
# create vector store
vector_store_name = "mini-llama-articles"
chroma_client = chromadb.PersistentClient(path=vector_store_name)
# Delete collection on rerun if already exists
if chroma_client.get_collection("mini-llama-articles"):
    chroma_client.delete_collection("mini-llama-articles")
chroma_collection = chroma_client.get_or_create_collection(vector_store_name)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
################################################################################

######################## LOAD THE ARTICLES ####################################
rows = []
# Load the file as a JSON
with open(join(dirname(dirname(__file__)), "mini-dataset.csv"), mode="r", encoding="utf-8") as file:
    csv_reader = csv.reader(file)

    for idx, row in enumerate(csv_reader):
        if idx == 0: continue  # Skip header row
        rows.append(row)
################################################################################

# Convert the chunks to Document objects so the LlamaIndex framework can process them.
documents = [Document(text=row[1], metadata={"title": row[0], "url": row[2], "source_name": row[3]}) for row in rows]

# Define the splitter object that splits the text into segments with 512 tokens,
# with a 128 overlap between the segments.
text_splitter = TokenTextSplitter(
    separator=" ", chunk_size=512, chunk_overlap=128
)

# Create the pipeline to apply the transformation on each chunk,
# and store the transformed text in the chroma vector store.
pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # If you are looking for a model that supports more languages, longer texts, and other retrieval methods, you can try using bge-m3.
    ],
    vector_store=vector_store
)

nodes = pipeline.run(documents=documents, show_progress=True)



######################## Load Vector Store and Create Query Engine ####################################

# Use the Together AI service to access the LLaMA2-70B chat model
llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key=os.environ["TOGETHER_AI_API_TOKEN"]
)

# create index from vector store
index = VectorStoreIndex.from_vector_store(vector_store, embed_model="local:BAAI/bge-small-en-v1.5")

# Define a query engine that is responsible for retrieving related pieces of text,
# and using a LLM to formulate the final answer.
query_engine = index.as_query_engine(llm=llm)

res = query_engine.query("How many parameters LLaMA 2 has?")
print(res.response)

for src in res.source_nodes:
    print("Node ID\\t", src.node_id)
    print("Title\\t", src.metadata['title'])
    print("Text\\t", src.text)
    print("Score\\t", src.score)
    print("-_"*20)

# Create questions for each segment.
llm = OpenAI(model="gpt-4o-mini")
rag_eval_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=1
)

######################## EVALUATION ####################################
rag_eval_dataset = generate_question_context_pairs(
        nodes,
        llm=llm,
        num_questions_per_chunk=1,
)
    
# Save the evaluation dataset as a json file for later use.
#rag_eval_dataset.save_json("./rag_eval_dataset_open_source_models.json")
rag_eval_dataset = EmbeddingQAFinetuneDataset.from_json(
    join(dirname(dirname(__file__)), "rag_eval_dataset_open_source_models.json")
)

async def run_evaluation(index, rag_eval_dataset, top_k_values, llm_judge,llm, n_queries_to_evaluate=20,num_work=1):
    evaluation_results = {}

    # ------------------- MRR and Hit Rate -------------------

    for top_k in top_k_values:
        # Get MRR and Hit Rate
        retriever = index.as_retriever(similarity_top_k=top_k)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )
        eval_results = await retriever_evaluator.aevaluate_dataset(rag_eval_dataset)
        avg_mrr = sum(res.metric_vals_dict["mrr"] for res in eval_results) / len(eval_results)
        avg_hit_rate = sum(res.metric_vals_dict["hit_rate"] for res in eval_results) / len(eval_results)

        # Collect the evaluation results
        evaluation_results[f"mrr_@_{top_k}"] = avg_mrr
        evaluation_results[f"hit_rate_@_{top_k}"] = avg_hit_rate

    # ------------------- Faithfulness and Relevancy -------------------

    # Extract the questions from the dataset
    queries = list(rag_eval_dataset.queries.values())
    batch_eval_queries = queries[:n_queries_to_evaluate]

    # Initiate the faithfulnes and relevancy evaluator objects
    faithfulness_evaluator = FaithfulnessEvaluator(llm=llm_judge)
    relevancy_evaluator = RelevancyEvaluator(llm=llm_judge)

    # The batch evaluator runs the evaluation in batches
    runner = BatchEvalRunner(
        {
            "faithfulness": faithfulness_evaluator,
            "relevancy": relevancy_evaluator
        },
        workers=num_work,
        show_progress=True,
    )

    # Get faithfulness and relevancy scores
    query_engine = index.as_query_engine(llm=llm)
    eval_results = await runner.aevaluate_queries(
        query_engine, queries=batch_eval_queries
    )
    faithfulness_score = sum(result.passing for result in eval_results['faithfulness']) / len(eval_results['faithfulness'])
    relevancy_score = sum(result.passing for result in eval_results['relevancy']) / len(eval_results['relevancy'])
    evaluation_results["faithfulness"] = faithfulness_score
    evaluation_results["relevancy"] = relevancy_score

    return evaluation_results

# We evaluate the retrievers with different top_k values.
top_k_values = [2, 4, 6, 8, 10]
llm_judge = OpenAI(temperature=0, model="gpt-4o")

# Use Llama 3.1 Model as inference LLM
evaluation_results = asyncio.run(run_evaluation(index, rag_eval_dataset, top_k_values, llm_judge,llm=llm,n_queries_to_evaluate=20,num_work=1))
print(evaluation_results)


# Use GPT-4o-mini as the LLM model
llm = OpenAI(model="gpt-4o-mini")

# run evaluation with GPT-4o-mini
top_k_values = [2, 4, 6, 8, 10]
llm_judge = OpenAI(temperature=0, model="gpt-4o")
evaluation_results = asyncio.run(run_evaluation(index, rag_eval_dataset, top_k_values, llm_judge,llm=llm, n_queries_to_evaluate=20,num_work=16))


######################## Inference Speed test ####################################

llm = TogetherLLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key=os.environ["TOGETHER_AI_API_TOKEN"]
)

time_start = time.time()
llm.complete("List the 50 states in the United States of America. Write their names in a comma-separated list and nothing else.")
time_end = time.time()
print("Time taken for Llama 3.1 70B on Together AI: {0:.2f} seconds".format(time_end - time_start))

llm = OpenAI(model="gpt-4o-mini")

time_start = time.time()
llm.complete("List the 50 states in the United States of America. Write their names in a comma-separated list and nothing else.")
time_end = time.time()
print("Time taken for GPT 4o Mini: {0:.2f} seconds".format(time_end - time_start))

llm = Gemini(model="models/gemini-1.5-flash")

time_start = time.time()
llm.complete("List the 50 states in the United States of America. Write their names in a comma-separated list and nothing else.")
time_end = time.time()
print("Time taken for Gemini 1.5 Flash: {0:.2f} seconds".format(time_end - time_start))

