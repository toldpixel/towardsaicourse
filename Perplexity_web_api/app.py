import os
import requests
import csv
import asyncio
import json
import pprint
from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.perplexity import Perplexity
from pydantic import BaseModel, Field
from llama_index.llms.perplexity import Perplexity
from llama_index.llms.openai import OpenAI
from ToolAnalysisWorkflow import ToolAnalysisWorkflow


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
os.environ["PERPLEXITY_API_KEY"] = os.environ.get("PERPLEXITY_API_KEY")
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]

# Google Sheets file URL (CSV export link)
url = 'https://docs.google.com/spreadsheets/d/1gHB-aQJGt9Nl3cyOP2GorAkBI_Us2AqkYnfqrmejStc/export?format=csv'

# Send a GET request to fetch the CSV file
response = requests.get(url)

# Initialize Perplexity LLM
perplexity_llm = Perplexity(
    api_key=os.environ["PERPLEXITY_API_KEY"],
    model="sonar", 
    temperature=0.2
)

# Initialize OpenAI LLM
openai_llm = OpenAI(model="gpt-4o", temperature=0.1) # gpt-4o-2024-08-06

response_list = []


# Check if the request was successful
if response.status_code == 200:
    # Decode the content to a string
    content = response.content.decode('utf-8')

    # Use the csv.DictReader to read the content as a dictionary
    csv_reader = csv.DictReader(content.splitlines(), delimiter=',')
    response_list = [row for row in csv_reader]
else:
    print(f"Failed to retrieve the file: {response.status_code}")

def create_questions_list(tool_name):
    questions_list = [
        f"Describe the developer tool <code>{tool_name}</code> in detail. What type of product is it, what types of tasks it is used for? What are the key features and capabilities of the tool? Explain Briefly",
        f"""
        Given the information about {tool_name}:
        <b>Parent Tools or languages</b>: Give a comma separated list of direct parent software languages or tools it uses. eg for Autokeras this would be; Keras, Python.
        <b>Languages it supports</b>: [List the programming languages {tool_name} this tool can be imported into, based on official documentation]
        Explain Briefly""",
        f"What other software tools are used in conjunction with {tool_name}. Explain Briefly",
        f"What company built and owns {tool_name} Is the product open source? Explain Briefly",
    ]
    return questions_list

def get_response_from_perplexity(question: str, llm):
    """
    Get a response from Perplexity LLM for a given question.

    Args:
        question (str): The question to ask.
        llm: The Perplexity LLM instance.

    Returns:
        The response from the LLM.
    """
    messages_dict = [
		    {"role": "system", "content": "Be Precise and concise"},
		    {"role": "user", "content": question},
    ]
    messages = [ChatMessage(**msg) for msg in messages_dict]
    response = llm.chat(messages)
    return response

def get_tool_response_perplexity(llm, tool_name):
    questions_list = create_questions_list(tool_name)
    response_list = []
    
    for question in questions_list:
        response = get_response_from_perplexity(question, llm)
        response_list.append({"question": question, "answer": str(response)})
    
    return response_list

async def get_structured_output(openai_llm, context):
    """
    Get structured output using Workflow instead of QueryPipeline.

    Args:
        openai_llm: The OpenAI LLM instance
        context: The context string to process

    Returns:
        Structured output from the workflow
    """
    workflow = ToolAnalysisWorkflow(openai_llm)
    result = await workflow.run(context=context)
    return result

response_dict = {}

async def process_tools():
    """Process all tools asynchronously."""
    for csv_dict in response_list[:5]:
        tool_name = csv_dict.get("Name")
        try:
            perplexity_response = get_tool_response_perplexity(perplexity_llm, tool_name)
            context = "\n".join(response.get("answer") for response in perplexity_response)

            structured_output = await get_structured_output(openai_llm, context)
            
            try:
              output_dict = json.loads(str(structured_output))
            except json.JSONDecodeError:
              output_dict = {"error": "Failed to parse structured output", "raw_output": str(structured_output)}

            response_dict[tool_name] = output_dict
            print(f"Processed {tool_name}")
        except Exception as e:
            print(f"Error processing {tool_name}: {e}")
            response_dict[tool_name] = {"error": str(e)}


asyncio.run(process_tools())

# Lets Visualize the result
for tool_name, tool_info in response_dict.items():
    print(f"Tool Name: {tool_name}")
    print(f"Description: {tool_info.get('description', 'N/A')}")
    pprint.pprint(tool_info)
    break

# Function to check if file exists
def file_exists(filename):
    return os.path.isfile(filename)

# Create a list to store the data for the CSV
data = []

# Iterate over the tool_output_dict
for tool_name, tool_info in response_dict.items():
    # Extract the relevant information
    description = tool_info.get("description", "")
    parent = tool_info.get("parent", "")
    languages = ", ".join(tool_info.get("languages", []))
    mapping = ", ".join(tool_info.get("mapping", []))
    category = ", ".join(tool_info.get("category", []))
    related_tools = ", ".join(tool_info.get("related_tools", []))
    company = tool_info.get("company", "")
    
    # Add the data for the current tool to the list
    data.append([tool_name, description, parent, languages, mapping, category, related_tools, company])

# Write or append the data to the CSV file
tool_output_filename = 'tool_output.csv'
file_exists_flag = file_exists(tool_output_filename)

with open(tool_output_filename, 'a' if file_exists_flag else 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header only if the file is being created for the first time
    if not file_exists_flag:
        writer.writerow(["Tool Name", "Description", "Parent", "Languages", "Mapping", "Category", "Related Tools", "Company"])
    
    # Append the data rows
    writer.writerows(data)