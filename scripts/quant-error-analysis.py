# %%
import pickle
from src.metrics.mimicsql_metrics import logic_form_accuracy_

from src.utils import query
DATABASE_LOCATION = "./data/TREQS/evaluation/mimic_db/mimic_all.db"

# %%
sql_eval_model = query(DATABASE_LOCATION)

# %%
filepath = "./results/mimicsql/run-11-cbr-cdb/cbr_cdb_results.pkl"

with open(filepath, "rb") as f:
    cbr_results = pickle.load(f)

cbr_results

# %%
filepath = "./results/mimicsql/run-11-rag-cdb/rag_cdb_results.pkl"

with open(filepath, "rb") as f:
    rag_results = pickle.load(f)

rag_results

# %%
logic_form_accuracy_(cbr_results, rag_results, sql_eval_model)

# %%
