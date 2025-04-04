# %%
import requests
import glob

# %%
response = requests.post(
    url="http://localhost:8000/query",
    json={
        "query": "",
    }
)

response.content

# %%
files = glob.glob("./data/Terveyskirjasto - Lääkärikirja Duodecim/*.xml")
documents = []

for id, file_path in enumerate(files):
    with open(file_path, "r", encoding="ISO-8859-1") as file:
        documents.append({
            "page_content": file.read(),
            "parent_id": str(id + 1),
            "source": str(file_path),
        })

documents

# %%
response = requests.post(
    url="http://localhost:8000/ingest", 
    json={
        "documents": documents[0:100],
    }
)

response.content

# %%
import sys
sys.path.append("app/")

from app.modules.retriever import QdrantRetriever
from app.modules.embedder import HuggingFaceEmbedder

embedder = HuggingFaceEmbedder()
retriever = QdrantRetriever(embedder=embedder)

# %%
retriever.retrieve("Aivohalvaus pähkinänkuoressa", 5)

# %%
retriever.reset()
