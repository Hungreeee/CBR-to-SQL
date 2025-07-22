# %%
from dotenv import load_dotenv
load_dotenv()

import json, pickle
from typing import List, Dict

from tqdm import tqdm

from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAG2SQL, CBR2SQL
from src.retriever import QdrantRetriever
from langchain_community.utilities.sql_database import SQLDatabase
from src.metrics import logic_form_accuracy, execution_accuracy

# %%
def train(rag_pipeline: RAG2SQL, dataset: List[Dict]):
    for data in tqdm(dataset, total=len(dataset)):
        query = data["question_refine"]
        sql_query = data["sql"]
        rag_pipeline.retain(query, sql_query)

def generate_results(rag_pipeline: RAG2SQL, dataset: List[Dict]):
    result_dataset = []

    for data in tqdm(dataset, total=len(dataset)):
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

# generator = OpenAIAgent()
generator = AzureAIAgent()

retriever = QdrantRetriever()
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
# # Train model
# train(rag_pipeline, trainset)

# %%
# Generate results for evaluation
# result_dataset_cbr = generate_results(cbr_pipeline, testset)
# result_dataset_cbr

# with open("./data/results/result_dataset_cbr2sql_gpt-4o_brittle.pkl", "wb") as f:
#     pickle.dump(result_dataset_cbr, f)

with open("./data/results/result_dataset_cbr2sql_gpt-4o_brittle.pkl", "rb") as f:
    result_dataset_cbr = pickle.load(f)

# %%
# # Generate results for evaluation
# result_dataset_rag = generate_results(rag_pipeline, testset)
# result_dataset_rag

# with open("./data/results/result_dataset_rag2sql_gpt-4o_brittle.pkl", "wb") as f:
#     pickle.dump(result_dataset_rag, f)

with open("./data/results/result_dataset_rag2sql_gpt-4o_topk3.pkl", "rb") as f:
    result_dataset_rag = pickle.load(f)

# %%
logic_form_accuracy(result_dataset_rag)

# %%
logic_form_accuracy(result_dataset_cbr)

# %%
# Compute metrics
ex_score = execution_accuracy(sql_db, result_dataset_cbr)
print(ex_score)
# %%
# Compute metrics
ex_score = execution_accuracy(sql_db, result_dataset_rag)
print(ex_score)

# %%
lf_scores = logic_form_accuracy(result_dataset_cbr)
print(lf_scores)

# %%
test_idx = 239
testset[test_idx]["question_refine"]

# %%
question = """
specify the number of patients who were admitted in the year less that 2187 and had  (aorto)coronary bypass of one coronary artery
"""

# %%
rag_pipeline.query(question)

# %%
masked_query, extracted_entities = cbr_pipeline.get_masked_question(question)
masked_query, extracted_entities

# %%
cbr_retriever.retrieve("what is the number of patients primarily diagnosed with condition?")

# %%
sql_template, retrieved_cases = cbr_pipeline.formulate_sql_template(question, masked_query, extracted_entities)
print(sql_template)

# %%
sql_query, relevant_entity_match = cbr_pipeline.discover_sources(question, sql_template, extracted_entities)
print(sql_query)

# %%
cbr_pipeline.lookup("mesothelial cells")

# %%
sql_db.run("""
what number of patients under the gae of 67 speak the language cape?
""")
