from huggingface_hub import hf_hub_download

file_path = hf_hub_download(repo_id="jaiganesan/ai_tutor_knowledge", filename="rag_research_paper.zip",repo_type="dataset",local_dir="./content")

