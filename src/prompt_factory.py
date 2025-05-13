entity_masking = """
Given an input text, your task is to substitute the entities that fit into the following categories with their corresponding entity type labels:
- CONDITION: This represents a clinical diagnosis or symptom documented in a patient's medical history.
- MEASUREMENT: This includes various clinical tests, assessments, or instruments.
- PROCEDURE: This refers to any intervention, surgical or non-surgical, that is performed on the patient for diagnostic or treatment purposes.
- DRUG: This refers to any therapeutic or prophylactic substance prescribed to a patient, including prescription medications, over-the-counter drugs, and vaccines.
- CODE: This refers to standardized medical codes, for example G71.038, N17.9, Z95.1, 92960
- TIME: This refers to any time-related information.
- PATIENT: This refers to the patient's identity.
- DEMOGRAPHY: This refers to the patient's demographic information (age, language, etc.).

Examples: 
Input text: How many females suffered from hypertension while taking venlafaxine?
Masked text: How many females suffered from CONDITION while taking DRUG?

Input text: Among the patients who had a Coronary Artery Bypass Grafting (CABG) surgery, as indicated by ICD-9-CM procedure codes (36.10 through 36.19) or ICD-10 code Z95.1, what proportion also had an Acute Kidney Injury (AKI) using ICD9 codes (584.0, 584.5, 584.6, 584.7, 584.8, 584.9, 586) and ICD10 code (N17.9)?
Masked text: Among the patients who had a Coronary Artery Bypass Grafting (CABG) surgery, as indicated by ICD-9-CM procedure codes (CODE through CODE) or ICD-10 code CODE, what proportion also had an Acute Kidney Injury (AKI) using ICD9 codes (CODE, CODE, CODE, CODE, CODE, CODE, CODE) and ICD10 code (CODE)?

Input text: How many females take antidepressants (citalopram, duloxetine) after myocardial infraction?
Masked text: How many females take DRUG_CLASS (DRUG, DRUG) after CONDITION?
"""

entity_extraction = """
Given an input text, your task is to extract the entities that fit into the following categories with their corresponding entity type labels:
- CONDITION: This represents any clinical diagnosis or symptom.
- MEASUREMENT: This includes various clinical tests, assessments, or instruments.
- DRUG: This refers to any therapeutic or prophylactic substance prescribed to a patient, including prescription medications, over-the-counter drugs, and vaccines.
- PATIENT: The name of the patient.
- CODE: Standardized medical codes.
"""

template_formulation = """
Given past examples of SQL queries and a question, revise the examples to create a SQL query that answers the question.
- Output only the raw SQL query—no markdown, explanations, or additional text.
- Strictly reuse tables, columns, conditions, and query structure from the provided SQL examples. Never guess, infer, or deviate unless the required elements are explicitly missing.
"""

source_discovery = """

"""

question_answering = """
Given the question and the retrieved data, address the given question.
- Always respond with the retrieved data, even if it does not make sense in terms of common sense (e.g., a patient has age 0).
"""

case_explanation = """
Explain how the SQL query is formulated concisely. For each value or condition in the SQL query, explain briefly its relation with the natural language query.
"""