# %%
from dotenv import load_dotenv
load_dotenv()

import json
from tqdm import tqdm
from typing import List, Dict
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAGtoSQL, CBRtoSQL
from src.retriever import QdrantRetriever
from langchain_community.utilities.sql_database import SQLDatabase

from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

np.random.seed(42)

# %%
DATABASE_URI = "sqlite:///./data/TREQS/evaluation/mimic_db/mimic_all.db"

# Collection names
RAG_CDB_COLLECTION = "rag_complete"      # RAG with all data
RAG_IDB_COLLECTION = "rag_incomplete"    # RAG with clustered data
CBR_CDB_COLLECTION = "cbr_complete"      # CBR with all data
CBR_IDB_COLLECTION = "cbr_incomplete"    # CBR with clustered data
LOOKUP_TABLE_COLLECTION = "lookup_table"

# Clustering parameters for IDB
MIN_CLUSTER_SIZE = 2
CLUSTER_EPSILON = 0.10

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

print(f"✓ Loaded {len(trainset)} training examples")
print(f"✓ Loaded {len(testset)} test examples")

# %%
sql_db = SQLDatabase.from_uri(DATABASE_URI)
generator = OpenAIAgent()
# generator = AzureAIAgent()  # Alternative

# Note: Lookup table should already be constructed using the lookup table script
lookup_table = QdrantRetriever(collection_name=LOOKUP_TABLE_COLLECTION)

# %%
# Initialize CBR-CDB pipeline
cbr_cdb_retriever = QdrantRetriever(collection_name=CBR_CDB_COLLECTION)
cbr_cdb_pipeline = CBRtoSQL(
    retriever=cbr_cdb_retriever,
    generator=generator,
    sql_db=sql_db,
    lookup_table=lookup_table,
)

# %%
cbr_cdb_pipeline.source_discovery(
    question="give the number of patients whose drug code is phen10i and lab test category is blood gas.",
    max_rounds=2
)

# %%
cbr_cdb_pipeline._extract_noun_phrases(
    question="give me the number of patients whose religion is christian scientist.",
    round_context={}
)

# %%
cbr_cdb_pipeline._lookup("delta", top_k=10)

# %%
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # %%
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # %%
# sentence1 = "patient name with diabetes staying more than 20 days"
# sentence2 = "patient name with diabetes staying less than a year"

# embedding1 = model.encode(sentence1)
# embedding2 = model.encode(sentence2)

# similarity = cosine_similarity([embedding1], [embedding2])[0][0]

# print(f"Similarity: {similarity:.4f}")

# %%
