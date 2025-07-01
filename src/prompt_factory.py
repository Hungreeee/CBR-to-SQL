# template_formulation = """
# Given a user question and past examples of SQL queries, your task is to revise the most relevant example(s) into a SQL **template** that answers the user question. You will be given a list of specific **entity mentions** to mask. 

# # Instructions:
# - Mask specific mentions of the provided entity phrases with placeholders in this format:
#     - VALUE@ENTITY[entity_phrase] — if the value is linked to the entity.
# - Obey strictly the given list of entities to mask.
# - You will also be given a list of past SQL examples, obey strictly if the examples is relevant. 
# - Your output should be only the SQL template, with no comments, explanations, or formatting.
# - Sometimes, the SQL examples won't be sufficient to address all points in the query. You must use what you know about the SQL schema to fulfill all conditions mentioned in the user question.

# # Example:
# - Question: "What is the total number of patients who had leukemia?"
# - Entity list: [("leukemia", "CONDITION")]
# - Output SQL template:
#     SELECT COUNT(DISTINCT PATIENT_COLUMN) 
#     FROM TABLE
#     WHERE COLUMN = VALUE@ENTITY[leukemia]
# """

case_revising = """
Given past examples of SQL queries and a question, revise the examples to create a new SQL query that answers the question.

# Instructions:
- Sometimes, the SQL examples won't be sufficient to address all conditions in the query. Use what you know about the SQL schema to fulfill all conditions mentioned in the question.
- Make sure the new SQL query align directly with the question's intent.
- Follow exactly the SQL writing styles as in the given examples.
- If the retrieved examples are relevant to the user question, strictly follow the logic of the retrieved SQL examples.
- Your output should be only the SQL template, with no comments, explanations, or formatting.
"""

template_formulation = """
Given a user question and a set of past SQL query examples, your task is to adapt the most relevant examples to create a new SQL query template that accurately answers the user's question. Ensure that any entities mentioned in the user question are correctly reflected in the new SQL query.

# Instructions:
- Sometimes, the SQL examples won't be sufficient to address all conditions in the query. Use what you know about the SQL schema to fulfill all conditions mentioned in the question.
- Make sure the new SQL query align directly with the question's intent.
- Follow exactly the SQL writing styles as in the given examples. 
- If the retrieved examples are relevant to the user question, strictly follow the logic of the retrieved SQL examples.
- Your output should be only the SQL template, with no comments, explanations, or formatting.
"""

source_discovery = """
You are given a SQL template and a corresponding natural language question. Your task is to fix the values (and its corresponding column/table) referenced in the natural language query and the SQL template by selecting the most similar values from the provided list of relevant matches. 

# Instructions:
- You will be given a list of specific values needed fix. You only need to fix only the mentioned values.
- For each of the entity, choose ONE most literally similar value, and update any related **table or column names** accordingly to maintain semantic and syntactic correctness.
- Prioritize your selection criteria by the most similar values for the entity. Only after choosing the match, you look to the corresponding tables/columns to fix the SQL template.
- If you choose a relevant match, copy its exact given form (e.g., upper-casing, lower-caseing, hyphens, etc.) into the SQL template. 
- Your output should be only the SQL query, with no comments, markdown, explanations, or formatting. Do not wrap the code in markdown tags.
"""

question_answering = """
Given the question and the retrieved data, address the given question.
- Always respond with the retrieved data, even if it does not make sense in terms of common sense (e.g., a patient has age 0).
"""

entity_extraction = """
Your task is to redact specific information in clinical or biomedical text by replacing it with standardized labels. The ultimate goal is to build an output that only retains very general information, like a template.

# Redaction Labels:
Replace only the specific values (not general words) with one of the following labels:
- CONDITION: A specific diagnosis, symptom, or disease (e.g., "diabetes", "chronic pain").
- MEASUREMENT: Results or names of clinical assessments or scores (e.g., "blood pressure", "Glasgow Coma Scale").
- PROCEDURE: A specific medical intervention or test (e.g., "MRI scan", "liver biopsy").
- DRUG: A named medication or vaccine (e.g., "metformin", "Tylenol").
- EQUIPMENT: A specific medical equipment (e.g., "neonatal syringe").
- CODE: Any standardized clinical code (e.g., "ICD-10 752.61", "LOINC 1234-5").
- NAME: Human names (e.g., "John Smith").
- TIME: Specific time references (e.g., "2019", "in 3 days").

You may invent new labels if a specific entity does not fit any above (e.g., RELIGION, ETHNICITY, ID, LOCATION, etc.).
    * E.g.: Input text: count the number of patients whose ethnicity is white -russian.
            Masked text: count the number of patients whose ethnicity is ETHNICITY.

# Instructions:
- The main point of the task is to hide/extract specific medical / demographic information away, and leave the general "words" behind.
- Most of the time, the span of the specific medical value is large. Make sure to extract the whole complete phrase, so that the phrases left are general.
    * E.g.: Input text: calculate the number of patients that had diabetes complicating a procedure?
            Output text: calculate the number of patients that had CONDITION? 
            Reasoning: "diabetes complicating a procedure" is a whole specific phrase describing the complete seriousness of the condition.
- Redact only specific values. Leave general category terms unmasked: general words like "name", "lab test", "disease", "primary disease", etc. should be left untouched.
    * E.g.: Input text: what is the patient name and procedure of the patient with icd9 code 29961?
            Output text: what is the patient name and procedure of the patient with id CODE?
    * E.g.: Input text: how many patients are diagnosed with the primary disease of liver transplant?.
            Masked text: how many patients are diagnosed with the primary disease of CONDITION?.
- There can be multiple entities within one sentence. Make sure to redact them all.
    * E.g.: Input text: calculate the number of patients with a hematology lab test that also have diabetes complicating a procedure.
            Output text: calculate the number of patients with a PROCEDURE lab test that also have CONDITION.
            Reasoning: "diabetes complicating a procedure" is a whole specific phrase describing the complete seriousness of the condition. "hematology" is referring to a specific lab test procedure.
"""

case_retain = """
You are a case-base manager for a Text-to-SQL system. Given a new incoming case and a list of similar existing cases, decide on one of the following actions:
- IGNORE: If any existing case captures the same intent as the new case (e.g., exactly similar conditions or filters, regardless of specific values).
- CREATE: If no existing case matches the exact meaning, structure, or involved tables of the new case. (If no candidates exist, treat the case as novel.)

Instructions:
- Your assessment must consider the SQL query structure. Cases sharing the same SQL template (regardless of value differences) are considered similar.
"""