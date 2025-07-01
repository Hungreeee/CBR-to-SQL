# %%
from dotenv import load_dotenv
load_dotenv()

import json, pickle
from tqdm import tqdm
from typing import List, Dict

from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAG2SQL, CBR2SQL
from src.retriever import QdrantRetriever
from langchain_community.utilities.sql_database import SQLDatabase

from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import re 
import numpy as np

np.random.seed(42)

#%%
def train(pipeline: RAG2SQL, dataset: List[Dict]):
    for data in tqdm(dataset):
        query = data["question_refine"]
        sql_query = data["sql"]
        pipeline.retain(query, sql_query)

def generate_results(rag_pipeline: RAG2SQL, dataset: List[Dict]):
    result_dataset = []

    for data in dataset:
        query = data["question_refine"]
        sql_query = data["sql"]

        try:
            response = rag_pipeline.query(query)
        except Exception as e:
            response = {
                "response": str(e),
                "sql_query": "",
                "sql_response": "",
                "retrieved_cases": "",
            }

        response.update({
            "query": query,
            "golden_sql_query": sql_query,
        })
        result_dataset.append(response)

    return result_dataset

# %%
trainset = []

with open("./data/TREQS/mimicsql_data/mimicsql_natural_v2/train.json", "r") as f:
    for line in f.readlines():
        json_object = json.loads(line)
        trainset.append(json_object)

testset = []

with open("./data/TREQS/mimicsql_data/mimicsql_natural_v2/test.json", "r") as f:
    for line in f.readlines():
        json_object = json.loads(line)
        testset.append(json_object)

with open("./data/TREQS/evaluation/mimic_db/lookup.json", "r") as f:
    lookup = json.load(f)

# %%
DATABASE_URI = "sqlite:///./data/TREQS/evaluation/mimic_db/mimic_all.db"
sql_db = SQLDatabase.from_uri(DATABASE_URI)

generator = OpenAIAgent()
# generator = AzureAIAgent()
retriever = QdrantRetriever(collection_name="rag_incomplete")

rag_pipeline = RAG2SQL(
    retriever=retriever,
    generator=generator,
    sql_db=sql_db,
)

lookup_table = QdrantRetriever(collection_name="lookup_table")
cbr_retriever = QdrantRetriever(collection_name="cbr_full")

cbr_pipeline = CBR2SQL(
    retriever=cbr_retriever,
    generator=generator,
    sql_db=sql_db,
    lookup_table=lookup_table,
)

# %%
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# %%
def remove_condition_values(sql):
    sql = re.sub(r'=\s*"[^"]*"', '= "VALUE"', sql)
    sql = re.sub(r'([<>]=?)\s*[\d\.]+', r'\1 VALUE', sql)         
    sql = re.sub(r'([<>]=?)\s*"[^"]*"', r'\1 "VALUE"', sql)    
    sql = re.sub(r'IN\s*\(([^)]*)\)', 'IN (VALUE)', sql, flags=re.IGNORECASE)
    return sql

# %%
encoded_sql = []

for idx, data in tqdm(enumerate(trainset), total=len(trainset)):
    embeddings = model.encode(remove_condition_values(data["sql"]))
    encoded_sql.append(embeddings)

# %%
hdb = HDBSCAN(
    min_cluster_size=2,
    n_jobs=5,
    cluster_selection_epsilon=0.10,
)
hdb.fit(encoded_sql)
labels = hdb.labels_

print(len(set(labels)))

# %%
label_dict = {}

for idx, label in enumerate(labels):
    if label not in label_dict:
        label_dict[label] = [trainset[idx]]
    else:
        label_dict[label].append(trainset[idx])

# %%

for label, items in sorted(label_dict.items(), key=lambda x: -len(x[1]), reverse=True):
    if label == -1:
        print(f"\n❌ Noise Cluster (-1), {len(items)} items:")
    else:
        print(f"\n✅ Cluster {label}, {len(items)} items:")

    for item in items[:10]: 
        print("   ", item)

    if len(items) > 10:
        print(f"   ... ({len(items) - 10} more)")

# %%
for label, items in sorted(label_dict.items(), key=lambda x: -len(x[1]), reverse=True):
    if label == -1:
        print(f"\n❌ Noise Cluster (-1), {len(items)} items:")
        for item in items: 
            rag_pipeline.retain(item["question_refine"], item["sql"])
    else:
        print(f"\n✅ Cluster {label}, {len(items)} items:")
        rag_pipeline.retain(items[0]["question_refine"], items[0]["sql"])

# %%
