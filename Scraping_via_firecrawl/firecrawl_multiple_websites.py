import requests
import csv
import pprint
import random
import os
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
import time
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings, VectorStoreIndex
from IPython.display import Markdown, display

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# Set up LLM, embedding model, and text splitter
llm = OpenAI(model="gpt-4o-mini")
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)

# Configure settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

"""
1. We have a Google Sheets document containing a list of AI tools, their descriptions, and URLs.
2. Our goal is to fetch this data and prepare it for web scraping using Firecrawl.
3. We’ll use the Python requests library to download the CSV file from Google Sheets.
4. Then, we’ll parse this CSV data using Python’s csv module.
5. Finally, we’ll prepare a subset of this data for our web scraping demonstration.
"""

# Google Sheets file URL (CSV export link)
url = 'https://docs.google.com/spreadsheets/d/1gHB-aQJGt9Nl3cyOP2GorAkBI_Us2AqkYnfqrmejStc/export?format=csv'

# Send a GET request to fetch the CSV file
response = requests.get(url)

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


# For demonstration purposes, we'll only crawl 10 websites
start_index = random.randint(0, len(response_list) - 3)
website_list = response_list[start_index:start_index+ 10] # Crawling 10 Websites only 

print("CSV data")
pprint.pprint(website_list)

# Crawl websites and handle responses
url_response = {}
crawl_per_min = 1  # Max crawl per minute for free version

# Track crawls
crawled_websites = 0

for i, website_dict in enumerate(website_list):
    url = website_dict.get('URL')
    print(f"Crawling: {url}")

    try:
        response = app.crawl_url(
            url,
            params={
                'limit': 5,  # Limit pages to scrape per site
                'scrapeOptions': {'formats': ['markdown', 'html']}
            }
        )
        crawled_websites += 1

        # Store the scraped data and associated info in the response dict
        url_response[url] = {
            "scraped_data": response.get("data"),
            "csv_data": website_dict
        }

    except Exception as exc:
        print(f"Failed to fetch {url} -> {exc}")
        continue

    # Pause to comply with crawl per minute limit
    if i != len(website_list) - 1 and (i + 1) % crawl_per_min == 0:
        print("Pausing for 1 minute to comply with crawl limit...")
        time.sleep(60)  # Pause for 1 minute after every crawl

documents = []

for url, scraped_content in url_response.items():
    csv_data = scraped_content.get("csv_data")
    scraped_results = scraped_content.get("scraped_data")

    for result in scraped_results:
        markdown_content = result.get("markdown")
        title = result.get("metadata").get("title")
        url = result.get("metadata").get("sourceURL")
        documents.append(
            Document(
                text=markdown_content,
                metadata={
                    "title": title,
                    "url": url,
                    "description": csv_data.get("Description"),
                    "category": csv_data.get("Category")
                }
            )
        )

# Create the index and query engine
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def display_response(response):
    display(Markdown(f"<b>{response}</b>"))

res = query_engine.query("What is Real-time multi-person keypoint detection")
display_response(res)

print("-----------------")
# Show the retrieved nodes
for src in res.source_nodes:
    print("Node ID\t", src.node_id)
    print("Title\t", src.metadata['title'])
    print("URL\t", src.metadata['url'])
    print("Score\t", src.score)
    print("Description\t", src.metadata.get("description"))
    print("Category\t", src.metadata.get("category"))
    print("-_"*20)