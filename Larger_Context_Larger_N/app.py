import os
import json
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import Settings
from huggingface_hub import hf_hub_download
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    KeywordExtractor,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline

load_dotenv()


############################# Preprocessing of AI Tutor creating the vectorstore.zip ###########################

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")

Settings.llm = Gemini(model="models/gemini-1.5-flash")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

file_path = hf_hub_download(repo_id="jaiganesan/ai_tutor_knowledge", filename="ai_tutor_knowledge.jsonl", repo_type="dataset", local_dir="./Larger_Context_Larger_N/content")

with open(file_path, "r") as file:
    ai_tutor_knowledge = [json.loads(line) for line in file]

def create_docs_from_list(data_list: List[dict]) -> List[Document]:
    documents = []
    for data in data_list:
        documents.append(
            Document(
                doc_id=data["doc_id"],
                text=data["content"],
                metadata={  # type: ignore
                    "url": data["url"],
                    "title": data["name"],
                    "tokens": data["tokens"],
                    "source": data["source"],
                },
                excluded_llm_metadata_keys=[
                    "title",
                    "tokens",
                    "source",
                ],
                excluded_embed_metadata_keys=[
                    "url",
                    "tokens",
                    "source",
                ],
            )
        )
    return documents

doc = create_docs_from_list(ai_tutor_knowledge)

# Define the splitter object that split the text into segments with 1536 tokens,
# with a 128 overlap between the segments.
text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)


# set up ChromaVectorStore and load in data
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("ai_tutor_knowledge")

# save to disk
db = chromadb.PersistentClient(path="/content/ai_tutor_knowledge")
chroma_collection = db.get_or_create_collection("ai_tutor_knowledge")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
llm = OpenAI(temperature=0, model="gpt-4o-mini")

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        QuestionsAnsweredExtractor(questions=2, llm=llm),
        SummaryExtractor(summaries=["prev", "self"], llm=llm),
        KeywordExtractor(keywords=10, llm=llm),
        OpenAIEmbedding(model = "text-embedding-3-small"),
    ],
    vector_store=vector_store,
)

# Run the transformation pipeline.
nodes = pipeline.run(documents=doc, show_progress=True)