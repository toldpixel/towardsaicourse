import newspaper
from llama_index.core.schema import Document

urls = [
    "https://docs.llamaindex.ai/en/stable/understanding",
    "https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/",
    "https://docs.llamaindex.ai/en/stable/understanding/indexing/indexing/",
    "https://docs.llamaindex.ai/en/stable/understanding/querying/querying/"
]

pages_content = []

# Retrieve the Content
for url in urls:
	try:
		article = newspaper.Article(url)
		article.download()
		article.parse()
		if len(article.text) > 0:
			pages_content.append({ "url": url, "title": article.title, "text": article.text })
	except:
		continue
	
print(pages_content[0])

# Convert the chunks to Document objects so the LlamaIndex framework can process them.
documents = [Document(text=row['text'], metadata={"title": row['title'], "url": row['url']}) for row in pages_content]