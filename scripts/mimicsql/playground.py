# %%
%load_ext autoreload
%autoreload 2

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

from src.utils import query
from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAGtoSQL, CBRtoSQL
from src.retriever import QdrantRetriever

from langchain_community.utilities.sql_database import SQLDatabase
from sentence_transformers import SentenceTransformer

np.random.seed(42)

# %%
DATABASE_URI = "sqlite:///./data/TREQS/evaluation/mimic_db/mimic_all.db"

# Collection names
RAG_CDB_COLLECTION = "rag_complete"     
RAG_IDB_COLLECTION = "rag_incomplete"
CBR_CDB_COLLECTION = "cbr_complete"     
CBR_IDB_COLLECTION = "cbr_incomplete"  

LOOKUP_COLLECTION = "lookup_table"

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
fallback_generator = OpenAIAgent()
generator = OpenAIAgent()  # Alternative

sql_eval_engine = query("./data/TREQS/evaluation/mimic_db/mimic_all.db")

# Note: Lookup table should already be constructed using the lookup table script
lookup_table = QdrantRetriever(collection_name=LOOKUP_COLLECTION)

# %%
# Initialize RAG-CDB pipeline
rag_cdb_retriever = QdrantRetriever(collection_name=RAG_CDB_COLLECTION)
rag_cdb_pipeline = RAGtoSQL(
    retriever=rag_cdb_retriever,
    generator=generator,
    sql_db=sql_db
)

# # Initialize CBR-CDB pipeline
# cbr_cdb_retriever = QdrantRetriever(collection_name=CBR_CDB_COLLECTION)
# cbr_cdb_pipeline = CBRtoSQL(
#     retriever=cbr_cdb_retriever,
#     generator=generator,
#     sql_db=sql_db,
#     lookup_table=lookup_table,
#     fallback_generator=fallback_generator
# )

# %%
# Provide the number of patients that died and had a primary disease of ST elevated myocardial infarction/cardiac cath.
# how many patients discharged to snf had hypoxia primary disease?

question = """
provide the number of private insurance patients who had incision of abdomen artery.
"""

# %%
rag_cdb_pipeline.handle_request(
    question=question,
)

# # %%
# rag_cdb_pipeline.handle_request(
#     question=question
# )

# %%
res = cbr_cdb_pipeline.source_discovery(
    question=question,
)

res

# %%
res_final = cbr_cdb_pipeline._construct_and_fill_sql(
    masked_question=res[0],
    original_question=question,
    entities=res[-1]
)

res_final

# %%
cbr_cdb_retriever.retrieve(
    query="how many patients were born before the year 2060?",
    top_k=5,
    hybrid=True
)

# %%
cbr_cdb_retriever.retrieve(
    query="how many patients were born before the year 2060?",
    top_k=5,
    hybrid=False
)

# %%
cbr_cdb_pipeline._lookup("Norepinephrine", top_k=5)

# %%
res = sql_db.run("""
SELECT DEMOGRAPHIC."ADMISSION_LOCATION", DEMOGRAPHIC."ADMISSION_TYPE" FROM DEMOGRAPHIC WHERE DEMOGRAPHIC."SUBJECT_ID" = "25543"
""")

# %%
pred_cur = sql_eval_engine.execute_sql("""
SELECT DEMOGRAPHIC."ADMISSION_LOCATION", DEMOGRAPHIC."ADMISSION_TYPE" FROM DEMOGRAPHIC WHERE DEMOGRAPHIC."SUBJECT_ID" = "25543"
""")
pred_out = pred_cur.fetchall()

pred_out

# %%
from collections import Counter

[tuple(sorted(row, key=str)) for row in pred_out]

# %%
