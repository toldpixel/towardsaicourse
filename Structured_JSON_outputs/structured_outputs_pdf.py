import base64
import os
import arxiv
import re
import json
from dotenv import load_dotenv
from pdf2image import convert_from_path
from instructions import system_instruction_prompt
from openai import OpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
client = OpenAI()

# The response format- JSON schema
json_response_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "research_paper_data",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "research_paper_data": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "source": { "type": "string" },
                "content": { "type": "string"}
            },

            "required": ["name", "source", "content"],
            "additionalProperties": False
          }
        },
      },
      "required": ["research_paper_data"],
      "additionalProperties": False
    }
  }
}


################################################### Extract the images ####################################################
pdf_directory = "./content/rag_research_paper"
output_dir = "./content/pages"
os.makedirs(output_dir, exist_ok=True)

pages_png = []

for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)

        convert = convert_from_path(pdf_path, use_pdftocairo=True)

        pdf_output_dir = os.path.join(output_dir, os.path.splitext(pdf_file)[0])
        os.makedirs(pdf_output_dir, exist_ok=True)

        for page_num, image in enumerate(convert):
            page_filename = f"page-{str(page_num + 1).zfill(3)}.png"
            full_path = os.path.join(pdf_output_dir, page_filename)
            image.save(full_path)

            pages_png.append(full_path)

print(pages_png)

################################################### Image Encoding ####################################################

"""
Next, the images—pages converted from a PDF—are encoded into base64 format. This encoding allows the images to be sent as input to the model through an API.
The primary challenge with extracting information from research papers and their graphs, 
architectures, tables, and charts to add context is parsing the PDF and analyzing it using the OpenAI Client. 
It is practical to convert the PDF to images, pass each page to the model, and extract information and data related to any graphs, 
architectures, tables, or charts. If the model identifies any graphs, architectures, tables, or charts, it can describe the data and the information they represent. 
The data or information is extracted with corresponding headlines or sub-headlines. 
"""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def arxiv_extraction(arxiv_id):
  client = arxiv.Client()

  search = arxiv.Search(id_list=re.findall(r'(\d{4}\.\d{5}|\w+(?:-\w+)?/\d{7})', arxiv_id), max_results=1)
  results = client.results(search)

  for result in results:
    return result.title, result.pdf_url

desc = []
for page in pages_png[:5]:
  # Getting the base64
  base64_image = encode_image(page)

  try:
    system_prompt = {"role": "system","content":system_instruction_prompt}
    user_prompt = {"role": "user","content": [{"type": "text", "text": "Please extract the content from this research paper page image."},
                                              {"type": "image_url", "image_url": {"url":f"data:image/jpeg;base64,{base64_image}","detail": "high"}}]}

		# OpenAI API Call
    response = client.chat.completions.create(
      model="gpt-4o-2024-08-06",
      temperature = 0,
      response_format=json_response_format,
      messages= [system_prompt,user_prompt],)

    if response.choices[0].message.content is None:
      continue

    result = json.loads(response.choices[0].message.content)

    if 'page-001' in page:
    # Or You can use the Image path to extract the Arxiv Research paper ID.
      research_paper_id = result['research_paper_data'][0]['source']
      
      # Arxiv API Call 
      research_paper_title,research_paper_url = arxiv_extraction(research_paper_id)
      for i in range(len(result['research_paper_data'])):
        result['research_paper_data'][i]['source'] = research_paper_id
        result['research_paper_data'][i]['name'] = research_paper_title +":"+ result['research_paper_data'][i]['name'] 
        result['research_paper_data'][i]['url'] = research_paper_url

    if 'page-001' not in page:
      for i in range(len(result['research_paper_data'])):
        result['research_paper_data'][i]['source'] = research_paper_id
        result['research_paper_data'][i]['name'] = research_paper_title +":"+ result['research_paper_data'][i]['name']
        result['research_paper_data'][i]['url'] = research_paper_url

    desc.extend(result['research_paper_data'])

  except Exception as e:
    print(response.choices[0].finish_reason)
    print(f"Skipping {page}... error: {e}")
    continue

print("Content research paper title and Headline :",desc[6]['name'],"\n")
print("Content :",desc[6]['content'],"\n")
print("Source :",desc[6]['source'],"\n")
print("URL :",desc[6]['url'])