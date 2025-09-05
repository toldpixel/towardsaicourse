import os
from openai import OpenAI
import json
from pydantic import BaseModel
from typing import List

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY1")

client = OpenAI()

# instead of defining a JSON schema for our “Top10BestSellingBooks” structure, we could use Pydantic like this
class Book(BaseModel):
    title: str
    author: str
    yearPublished: int
    summary: str

class Top10BestSellingBooks(BaseModel):
    books: List[Book]


# The response format-JSON schema
response_format_json = {
  "type": "json_schema",
  "json_schema": {
    "name": "Top10BestSellingBooks",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "Top10BestSellingBooks": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "title": { "type": "string" },
              "author": { "type": "string" },
              "yearPublished": { "type": "integer" },
              "summary": { "type": "string" }
            },
            "required": ["title", "author", "yearPublished", "summary"],
            "additionalProperties": False
          }
        }
      },
      "required": ["Top10BestSellingBooks"],
      "additionalProperties": False
    }
  }
}

system_prompt = """
You are a helpful assistant designed to output information exclusively in JSON format.
### Example JSON Format
{
  "Top10BestSellingBooks": [
    {
      "title": "Book Title",
      "author": "Author Name",
      "yearPublished": "Year",
      "summary": "Brief summary of the book."
    }
  ]
}

"""

prompt = "Give me the names of the 10 best-selling books, their authors, the year they were published, and a concise summary in JSON format"


# API Call
response = client.chat.completions.create(
  model="gpt-4o-2024-08-06",
  response_format=Top10BestSellingBooks, #response_format_json,
  temperature = 0,
  messages=[
    {"role": "system", "content":system_prompt},
    {"role": "user", "content": prompt}
  ]
)

result_book = json.loads(response.choices[0].message.content)

print(result_book['Top10BestSellingBooks'][0])
print("-------------------------------")
print(result_book['Top10BestSellingBooks'][0]['title'])

