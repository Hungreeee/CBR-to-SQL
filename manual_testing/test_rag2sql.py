# %%
import os
import ast
import pandas as pd

from langchain_ollama import ChatOllama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

# %%
DATABASE_URI = "mysql+pymysql://root:12345678@localhost:3306/mysql"
db = SQLDatabase.from_uri(DATABASE_URI)

# %%
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_ctx=10000,
)

# %%
df = pd.read_csv("data/Mock Pheno Data/pre-diabetic_data.csv")
df.to_sql("diabetes_patient_tracker", con=db._engine, if_exists="replace", index=False)

db = SQLDatabase.from_uri(DATABASE_URI)
db.get_table_names()

# %%
template = """
Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Instructions:
- Return only the SQL query, do not include any other text data. 

Question: {question}
SQL Query:
"""

prompt = ChatPromptTemplate.from_template(template)

# %%
sql_chain = (
    RunnablePassthrough.assign(
        schema=lambda _: db.get_table_info(["diabetes_patient_tracker"])
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# %%
user_question = 'How many meals have I had?'
response = sql_chain.invoke({"question": user_question})
print(response)

# %%
db.run("""
SELECT SUM(`protein (g)`) 
FROM diabetes_patient_tracker 
WHERE timestamp >= '12:00:00' AND timestamp < '13:00:00';
""")

# %%
print(db.get_table_info(["diabetes_patient_tracker"]))

# %%
template = """
Based on the table schema below, question, and sql response, write a natural language response to the question:
{schema}
Question: {question}
Retrieved data: {response}
"""

prompt_response = ChatPromptTemplate.from_template(template)

full_chain = (
    RunnablePassthrough
    .assign(query=sql_chain)
    .assign(
        schema=lambda _: db.get_table_info(["diabetes_patient_tracker"]),
        response=lambda vars: db.run(vars["query"]),
    )
    | prompt_response
    | llm
)

# %%
user_question = "What did I have in my latest meal?"
response = full_chain.invoke({"question": user_question}, config={'callbacks': [ConsoleCallbackHandler()]})
print(response.content)

# %%
user_question = "What is the total protein I have consumsed during 12-13 o'clock?"
response = full_chain.invoke({"question": user_question}, config={'callbacks': [ConsoleCallbackHandler()]})
print(response.content)

# %%