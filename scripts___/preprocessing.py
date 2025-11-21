# %%
import json
import pandas as pd
import glob
from langchain_community.utilities.sql_database import SQLDatabase

# %%
json_objects = []

with open("./data/TREQS/mimicsql_data/mimicsql_natural_v2/test.json", "r") as f:
    for line in f.readlines():
        json_object = json.loads(line)
        json_objects.append(json_object)

json_objects

# %%
DATABASE_URI = "sqlite:///./data/TREQS/evaluation/mimic_db/mimic_all.db"
db = SQLDatabase.from_uri(DATABASE_URI)

db.get_table_info()

# %%
db.run(json_objects[0]["sql"])
