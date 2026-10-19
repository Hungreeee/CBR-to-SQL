# %%
"""
Case Retention Script for the semantic-tagging-only ablation (Interactive)
Builds a separate CBR-to-SQL CDB collection with refined_tagging=False:
questions are masked with the generic tags from entity extraction. The lookup
table still resolves each entity's best match, table and column.
"""

from dotenv import load_dotenv
load_dotenv()

import json
from tqdm import tqdm

from src.configs import RAGConfig, LLMConfig
from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import CBRtoSQL
from src.retriever import QdrantRetriever
from langchain_community.utilities.sql_database import SQLDatabase

# ========== CONFIGURATION ==========

# %%
DATABASE_URI = "sqlite:///./data/TREQS/evaluation/mimic_db/mimic_all.db"

# Collection names
CBR_CDB_SEM_RET_COLLECTION = "cbr_complete_sem_ret"
LOOKUP_TABLE_COLLECTION = "lookup_table"

# Generator model, pinned here so the build does not follow the LLMConfig default
MODEL = "gpt-4.1-mini"

# ========== LOAD DATA ==========

# %%
print("\n=== Loading Datasets ===")

trainset = []
with open("./data/TREQS/mimicsql_data/mimicsql_natural_v2/train.json", "r") as f:
    for line in f.readlines():
        json_object = json.loads(line)
        trainset.append(json_object)

print(f"✓ Loaded {len(trainset)} training examples")

# ========== INITIALIZE COMPONENTS ==========

# %%
print("\n=== Initializing Components ===")

sql_db = SQLDatabase.from_uri(DATABASE_URI)
llm_config = LLMConfig(model=MODEL)
generator = OpenAIAgent(llm_config)
# generator = AzureAIAgent(llm_config)  # Alternative
print(f"✓ Generator: {type(generator).__name__} ({generator.config.model})")

lookup_table = QdrantRetriever(collection_name=LOOKUP_TABLE_COLLECTION)

# ========== CDB (COMPLETE DATABASE), SEMANTIC TAGGING ONLY ==========

# %%
print("\n--- CBR-to-SQL CDB (semantic tagging only) ---")
cbr_cdb_retriever = QdrantRetriever(collection_name=CBR_CDB_SEM_RET_COLLECTION)
cbr_cdb_retriever.reset()
cbr_cdb_pipeline = CBRtoSQL(
    retriever=cbr_cdb_retriever,
    generator=generator,
    sql_db=sql_db,
    lookup_table=lookup_table,
    config=RAGConfig(dataset="mimicsql", refined_tagging=False),
)

# %%
# Retain all training cases (generic entity tagging only)
print(f"Retaining {len(trainset)} cases for CBR-CDB (semantic tagging only)...")
for data in tqdm(trainset, desc="CBR-CDB sem-ret retention"):
    cbr_cdb_pipeline.retain_case(data["question_refine"], data["sql"])
print("✓ CBR-CDB (semantic tagging only) case retention complete!")

# %%
