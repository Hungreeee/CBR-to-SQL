# %%
from dotenv import load_dotenv
load_dotenv()

import json, pickle
from typing import List, Dict

from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAG2SQL
from src.retriever import QdrantRetriever
from langchain_community.utilities.sql_database import SQLDatabase
from src.metrics import logic_form_accuracy, execution_accuracy

#%%
def train(rag_pipeline: RAG2SQL, dataset: List[Dict]):
    for data in dataset:
        query = data["question_refine"]
        sql_query = data["sql"]
        rag_pipeline.retain(query, sql_query)

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
retriever = QdrantRetriever()

rag_pipeline = RAG2SQL(
    retriever=retriever,
    generator=generator,
    sql_db=sql_db,
)

# %%
# train(rag_pipeline, trainset)

# %%
# result_dataset = generate_results(rag_pipeline, testset)
# result_dataset

# with open("./data/results/result_dataset_rag2sql.pkl", "wb") as f:
#     pickle.dump(result_dataset, f)

with open("./data/results/result_dataset_rag2sql.pkl", "rb") as f:
    result_dataset = pickle.load(f)

# %%
ex_score = execution_accuracy(sql_db, result_dataset)
ex_score

# %%
lf_scores = logic_form_accuracy(lookup, result_dataset)
lf_scores

# %%
logic_form_accuracy(lookup, [{
    "sql_query": 'select lab."itemid", lab."flag" from lab where lab."subject_id" = "22377"',
    "golden_sql_query": 'select lab."itemid", lab."flag" from lab where lab."subject_id" = "22377"',
}])

# %%
with open("./data/TREQS/evaluation/generated_sql/output.json", 'w', encoding='utf-8') as f:
    for item in result_dataset:
        entry = {
            "sql_pred": item["sql_query"],
            "sql_gold": item["golden_sql_query"]
        }
        f.write(json.dumps(entry) + '\n')

# %%
test_idx = 879
testset[test_idx]

# %%
response = rag_pipeline.query(testset[test_idx]["question_refine"])
response

# %%
second_results = sql_db.run(testset[test_idx]["sql"])
second_results

# %%
response["sql_response"]

# %%
# Execution Accuracy
response["sql_response"] == second_results

# %%
testset[test_idx]["sql"]

# %%
response["sql_query"]

# %%
retriever.retrieve("heparin imw")

# %%
import numpy as np

for i in list(np.where(np.array(scores) == 0)[0]):
    print(i)
    print(result_dataset[i])
    print()
