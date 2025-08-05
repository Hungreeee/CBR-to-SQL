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

# with open("./data/results/result_dataset_cbr2sql_gpt-4o_incomplete_brittle.pkl", "wb") as f:
#     pickle.dump(result_dataset_cbr, f)

with open("./data/results/result_dataset_cbr2sql_gpt-4o.pkl", "rb") as f:
    result_dataset_cbr = pickle.load(f)

# %%
# # Generate results for evaluation
# result_dataset_rag = generate_results(rag_pipeline, testset)
# result_dataset_rag

# with open("./data/results/result_dataset_rag2sql_gpt-4o_incomplete_brittle.pkl", "wb") as f:
#     pickle.dump(result_dataset_rag, f)

with open("./data/results/result_dataset_rag2sql_gpt-4o.pkl", "rb") as f:
    result_dataset_rag = pickle.load(f)

# %%
logic_form_accuracy(result_dataset_cbr)

# %%
logic_form_accuracy(result_dataset_rag)

# %%
# Compute metrics
ex_score, error_cbr, success_cbr = execution_accuracy(sql_db, result_dataset_cbr)
print(ex_score)

# %%
# Compute metrics
ex_score, error_rag, success_rag = execution_accuracy(sql_db, result_dataset_rag)
print(ex_score)

# %%
selected_set = [i for i in success_rag if i in error_cbr]
cbr_fail = [j for i, j in enumerate(result_dataset_cbr) if i in selected_set]

# %%
rag_success = [j for i, j in enumerate(result_dataset_rag) if i in selected_set]

# %%
for i, j in zip(cbr_fail, rag_success):
    print("Question:", i["query"])
    print("CBR SQL (Fail):", i["sql_query"])
    print("RAG SQL (Success):", j["sql_query"])
    print("\n-----\n")

# %%
import matplotlib.pyplot as plt

# Data
top_k = ['@top-1', '@top-3', '@top-5']
rag_scores = [0.756, 0.787, 0.811]
cbr_scores = [0.775, 0.806, 0.828]

rag_scores_ = [0.789,0.831 , 0.855]
cbr_scores_ = [0.836, 0.863, 0.882]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Plot lines
ax[1].plot(top_k, cbr_scores, marker='o', label='CBR-to-SQL')
ax[1].plot(top_k, rag_scores, marker='o', label='RAG-to-SQL')
ax[1].set_xlabel('k-shot')
ax[1].set_ylabel('Logical Form Accuracy')
ax[1].set_title('Logical form accuracy on varying level of top-k')
ax[1].set_ylim((0.75, 0.89))
ax[1].grid(True, linestyle='--', alpha=0.5)
fig.legend(loc=(0.35, 0.), ncol=2, labelspacing=0.)

# Plot lines
ax[0].plot(top_k, cbr_scores_, marker='o', label='CBR-to-SQL')
ax[0].plot(top_k, rag_scores_, marker='o', label='RAG-to-SQL')
ax[0].set_xlabel('k-shot')
ax[0].set_ylabel('Execution Accuracy')
ax[0].set_title('Execution accuracy on varying level of top-k')
ax[0].set_ylim((0.75, 0.89))
ax[0].grid(True, linestyle='--', alpha=0.5)
# ax[1].legend()

plt.show()

# %%
test_idx = 239
testset[test_idx]["question_refine"]

# %%
question = """
What is the average age of patients whose language is cape and primary disease is
hyperglycemia?
"""

# %%
cbr_pipeline.query(question)

# %%
cbr_pipeline.lookup("csf", top_k=100)

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
