import os
import asyncio
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import glob

# API access to llama-cloud
load_dotenv()
os.environ["LLAMA_CLOUD_API_KEY"] = os.environ.get("LLAMA_CLOUD_API_KEY")
#file_path = hf_hub_download(repo_id="jaiganesan/ai_tutor_knowledge", filename="research_papers_llamaparse.zip",repo_type="dataset",local_dir="./content")


#Parser
parser = LlamaParse(
    result_type="markdown",
    verbose=True,
)

data_dir="./content/research_papers_llamaparse"
file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

documents = asyncio.run(parser.aparse(file_paths))
print(documents[0].pages[0].text)

pdf_files = glob.glob("./content/research_papers_llamaparse/*.pdf")
parser = LlamaParse(verbose=True)
json_objs=parser.get_json_result(pdf_files)

print(json_objs[0])