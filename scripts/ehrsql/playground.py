# %%
%load_ext autoreload
%autoreload 2

# %%
from dotenv import load_dotenv
load_dotenv()

import json
import numpy as np

from src.configs import RAGConfig
from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAGtoSQL, CBRtoSQL
from src.retriever import QdrantRetriever
from src.utils import query
from src.metrics import parse_sql, parse_sql_

from langchain_community.utilities.sql_database import SQLDatabase

np.random.seed(42)

# %%
DATABASE_URI = "sqlite:///./data/EHRSQL/dataset/ehrsql/mimic_iii/mimic_iii.sqlite"
DATABASE_LOCATION = "./data/EHRSQL/dataset/ehrsql/mimic_iii/mimic_iii.sqlite"

# Collection names
RAG_CDB_COLLECTION = "rag_complete_ehrsql"      # RAG with all data
RAG_IDB_COLLECTION = "rag_incomplete_ehrsql"    # RAG with clustered data
CBR_CDB_COLLECTION = "cbr_complete_ehrsql"      # CBR with all data
CBR_IDB_COLLECTION = "cbr_incomplete_ehrsql"    # CBR with clustered data

LOOKUP_COLLECTION = "lookup_table_ehrsql"

# Clustering parameters for IDB
MIN_CLUSTER_SIZE = 2
CLUSTER_EPSILON = 0.10

# %%
trainset = []
with open("./data/EHRSQL/dataset/ehrsql/mimic_iii/train.json", "r") as f:
    trainset = json.loads(f.read())

testset = []
with open("./data/EHRSQL/dataset/ehrsql/mimic_iii/test.json", "r") as f:
    testset = json.loads(f.read())

print(f"✓ Loaded {len(trainset)} training examples")
print(f"✓ Loaded {len(testset)} test examples")

# %%
sql_db = SQLDatabase.from_uri(DATABASE_URI)
fallback_generator = OpenAIAgent()
generator = AzureAIAgent()  # Alternative
sql_eval_model = query(DATABASE_LOCATION)

# Note: Lookup table should already be constructed using the lookup table script
lookup_table = QdrantRetriever(collection_name=LOOKUP_COLLECTION)

# %%
# rag_cdb_retriever = QdrantRetriever(collection_name=RAG_CDB_COLLECTION)
# rag_cdb_pipeline = RAGtoSQL(
#     retriever=rag_cdb_retriever,
#     generator=generator,
#     sql_db=sql_db,
#     config=RAGConfig(dataset="ehrsql")
# )

# Initialize CBR-CDB pipeline
cbr_cdb_retriever = QdrantRetriever(collection_name=CBR_CDB_COLLECTION)
cbr_cdb_pipeline = CBRtoSQL(
    retriever=cbr_cdb_retriever,
    generator=generator,
    sql_db=sql_db,
    lookup_table=lookup_table,
    fallback_generator=fallback_generator,
    config=RAGConfig(dataset="ehrsql")
)

# %%
question = """
what's the name of the med that shouldn't be given during exc/dest hrt lesion open.
"""

# %%
cbr_cdb_pipeline.handle_request(
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
cbr_cdb_pipeline._lookup("echinococc granul nos", top_k=5)

# %%
cbr_cdb_retriever.retrieve(
    query=res[0]
)

# %%
cbr_cdb_retriever.retrieve(
    query="when did patient 1372 since 102 months ago last get a mrsa screen microbiology test?",
    top_k=100
)

# %%
sql_db.run("""
select count( distinct t1.subject_id ) from ( select admissions.subject_id, diagnoses_icd.charttime from diagnoses_icd join admissions on diagnoses_icd.hadm_id = admissions.hadm_id where diagnoses_icd.icd9_code = ( select d_icd_diagnoses.icd9_code from d_icd_diagnoses where d_icd_diagnoses.short_title = 'myelodysplastic synd nos' ) and datetime(diagnoses_icd.charttime) >= datetime(current_time,'-3 year') ) as t1 join ( select admissions.subject_id, procedures_icd.charttime from procedures_icd join admissions on procedures_icd.hadm_id = admissions.hadm_id where procedures_icd.icd9_code = ( select d_icd_procedures.icd9_code from d_icd_procedures where d_icd_procedures.short_title = 'cont inv mec ven <96 hrs' ) and datetime(procedures_icd.charttime) >= datetime(current_time,'-3 year') ) as t2 on t1.subject_id = t2.subject_id where t1.charttime < t2.charttime and datetime(t2.charttime) between datetime(t1.charttime) and datetime(t1.charttime,'+2 month')
""")

# %%
# db_head = sql_eval_model.db_head

# headerDic = []
# for tb in db_head:
#     for hd in db_head[tb]:
#         headerDic.append('.'.join([tb, hd]).lower())

# tableDic = []
# for tb in db_head:
#     tableDic.append(tb.lower())

# # %%
# parse_sql_("""
# select microbiologyevents.org_name from microbiologyevents where microbiologyevents.hadm_id in ( select admissions.hadm_id from admissions where admissions.subject_id = 28956 and admissions.dischtime is not null order by admissions.admittime asc limit 1 ) and microbiologyevents.spec_type_desc = 'bronchial washings' and microbiologyevents.org_name is not null order by microbiologyevents.charttime asc limit 1
# """,
# headerDic, tableDic)