# %%
from src.retriever import QdrantRetriever
from data.TREQS.evaluation.utils import *
from langchain_community.utilities.sql_database import SQLDatabase

# %%
DATABASE_URI = "sqlite:///./data/EHRSQL/dataset/ehrsql/mimic_iii/mimic_iii.sqlite"
sql_db = SQLDatabase.from_uri(DATABASE_URI)

# %%
db_file = 'data/TREQS/evaluation/mimic_db/mimic.db'
model = query(db_file)
(db_meta, db_tabs, db_head) = model._load_db(db_file)

# %%
lookup_table = QdrantRetriever(collection_name="lookup_ehrsql")

# %%
datapoints = []
ignored_columns = ["TIME", "DOB", "DOD", "CODE", "DRUG_DOSE", "VALUE_UNIT"]

for tb in db_meta:
    for hd in db_meta[tb]:
        mysql = 'select distinct {} from {}'.format(hd, tb)
        myres = model.execute_sql(mysql).fetchall()
        myres = list({k[0]: {} for k in myres if not k[0] == None})
        db_meta[tb][hd] = myres

        for entity in myres:
            if isinstance(entity, str):
                if not any([x in hd for x in ignored_columns]):
                    datapoints.append({
                        "entity": entity,
                        "table": tb,
                        "column": hd,
                    })

datapoints

# %%
lookup_table.ingest(datapoints, "entity")

# %%
lookup_table.retrieve("diphenhydramine", 50)

# %%
