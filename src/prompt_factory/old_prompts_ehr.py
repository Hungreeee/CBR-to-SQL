# ========== RAG-to-SQL Prompts ==========

case_revising = """
Generate a SQL query by adapting retrieved examples to answer the original natural language question. Return "None" if insufficient information is provided to write a SQL query.

# Procedure
1. Select the most semantically aligned retrieved examples based on semantic intent.
2. Treat the selected examples as the authoritative template, and exactly follow its formulation.
3. Generate the final SQL query by reusing from selected similar example's logic, structure, and formatting. You have the option to return "None", if there is insuffient context to write a SQL query (impossible question).

# Rules

## 1. Example Selection Rule
1. Select the examples with the most aligned semantic meaning and logic. Then, Copy that examples' SQL queries' structure to adapt to the current questionc
2. There is always one filter clause for a condition mentioned in the original question. Do not attempt to address one condition value mutliple times.

## 2. IMPORTANT - Impossible Question Detection
Some questions are intentionally impossible to answer with the available data. Before writing SQL, verify that the question's logical form is fully supported by the retrieved examples and schema.

### Return `"None"` if ANY are true:
* Retrieved examples do **not** demonstrate how to answer **all semantic aspects** of the question's logical form.
* The schema lacks required tables or columns for **any** part of the question.
* You cannot confidently map **all question components** to database elements.

### Verification checklist (must pass all):
* Do retrieved examples share the **same logical form**, even with different entities?
* Does the schema contain **all required tables and columns**?
* Can **every condition and requirement** be expressed in SQL using the provided information?

### Important
* If retrieved examples share the same SQL structure and differ only by entity values, the question **IS answerable**.

## 3. Output Constraints
* Return "None" if the input question is impossible to be answered.
* Otherwise, output ONLY one SQL query, no markdown, comments, explanations, or extra text. Closely the writing style of the retrieved SQLs (no linebreaks, etc.)
* Exactly one value per entity placeholder.
* Write SQL minimally, while strictly following the type, formatting, and logical conventions of the retrieved example.
"""
 
# ========== CBR-to-SQL: Source Discovery - Iterative Refinement ==========

entity_extraction = """
You are performing clinical database normalization.

Your task is to identify spans of text that correspond to standardized database concepts (e.g., procedure names, diagnostic labels, or coded attributes).

This is a text alignment task only. 
- Do not infer causes, outcomes, or intent. 
- Do not reason about events. 
- Only return text spans that match database concepts.

For each extracted entity, you must:
1. Extract the complete value span (preserve typos, special characters, parentheses, slashes) while ignoring generic terms (given as a list)
2. Assign a temporary tag based on sentence context
3. Flag whether it's semantic (needs database lookup) or structural (use as-is)

# Entity Types:

SEMANTIC ENTITIES (is_semantic = true):
Examples of entities that need database lookup validation:
- CONDITION: Specific diagnosis, disease (e.g., "diabetes", "sepsis")
- PROCEDURE: Specific intervention or test (e.g., "MRI scan", "cardiac catheterization")
- DRUG: Named medication (e.g., "metformin", "sonata", "impact")
- MEASUREMENT: Lab test names (e.g., "creatinine", "blood pressure")
- EQUIPMENT: Medical devices (e.g., "ventilator")
- ETHNICITY: Ethnic group (e.g., "Hispanic", "African American")
- RELIGION: Religion (e.g., "Christian Scientist", "Catholic")
- GENDER: Gender value (e.g., "Male", "Female")
- LANGUAGE: Language (e.g., "Spanish", "English")
- LOCATION: Specific places (e.g., "ICU", "Emergency Department")
- NAME: Human names (e.g., "John Smith")

STRUCTURAL ENTITIES (is_semantic = false):
Examples of values that are used as-is for filtering:
- ID: Patient identifiers (e.g., "12345", "SUBJ_001")
- ICD_CODE: Diagnosis codes (e.g., "29961", "V45.81", "250.00")
- AGE_VALUE: Age numbers (e.g., "65", "18")
- YEAR: Year values (e.g., "2019", "2150")
- NUMERIC_VALUE: Other numeric values (e.g., "2.0", "100")

# Extraction Rules:
1. IMPORTANT: **Do not extract generic words:**
   - List of generic keywords to IGNORE during EXTRACTION: "patient", "microbiology test", "lab test", "procedure", "drug route", "route of drug administration", "hospital admission", and similar terms.
      - "when did patient 3369 until 67 months ago last get a viral culture:r/o herpes simplex virus microbiology test?" → extract "viral culture:r/o herpes simplex virus", leaving "microbiology test" because it is clearly similar to the generic terms.
2. **Extract COMPLETE medical phrases, but exclude generic terms**
   - Include everything: slashes (/), parentheses (), hyphens (-)
      - "viral culture:r/o herpes simplex virus" → Extract ENTIRE phrase
      - "promote w/fiber" → Extract COMPLETE phrase
   - Exclude generic terms:
      - "did there exist any organism found in the first peripheral blood lymphocytes microbiology test of patient 2957 during this hospital visit?" → extract "peripheral blood lymphocytes", exclude "first" and "microbiology test", which is generic terms.
3. **Preserve original form:**
   - Keep typos, capitalization, spacing, special characters exactly as written
   - "lipalse" → "lipalse" (not "lipase")
4. **Tagging decision:**
   - Medical/demographic term that might have variations → is_semantic = true
   - ID/number/code for exact matching → is_semantic = false
5. **Compound Entity Extraction**
   - Extract compound entities as a SINGLE span when connected by conjunctions (and/or/but/not) or punctuation (/, ;, comma).
   - Do NOT split into separate entities - capture the entire phrase.
   Examples:
   - "hypertension but not coronary artery disease" → Extract: "hypertension but not coronary artery disease" ✓
   - "diabetes and sepsis" → Extract: "diabetes and sepsis" ✓
   - "aspirin/ibuprofen" → Extract: "aspirin/ibuprofen" ✓
   Do NOT extract as separate entities:
   - ✗ "hypertension", "coronary artery disease"
   - ✗ "diabetes", "sepsis"
   - ✗ "aspirin", "ibuprofen"
"""

tag_assignment = """
You are given a an entity value. Your task is to select one highest matching value from the list of real database entities OR reject all if no good match exists.

# Instructions:
- Evaluate candidates using lexical similarity as the highest priority, and semantic correctness as a secondary validation.
   - If a good match exists: Select the highest word overlap (considering also word ordering and special characters as needed) by choosing its index and derive tag from its table/column.
   - If NO good match exists: Set best_match_index = -1 and label = "NO_MATCH"
   Select the candidate with the highest surface-form similarity (word-level or character-level).
   - Consider word overlap, word order, abbreviations, truncations, and special-character variations. Do NOT prefer longer, clearer, or more descriptive terms. Do NOT expand abbreviations or normalize for readability. If a shorter or abbreviated term has the highest score, it must be selected, even if a longer term appears semantically clearer.

Choose its index and derive the label from its table/column.
- The second highest priority is to verify semantic correctness:
   - Does the mapped value match the entity's actual meaning? If not, reject criteria (index = -1).
   - Example: "dead" → EXPIRE_FLAG = 1 ✓ CORRECT | "dead" → "DAH" ✗ WRONG (unrelated)
- Table/column preference can be used as a tie-breaker, but **must not override lexical priority**. However, if the question context specifies a particular type of query, use it to guide selection among equally close matches. 

# Accept criteria:
- IMPORTANT: Always prefer the term with the highest word-level or character-level similarity, or with substantial word overlap (including word order and special-character variations). Do NOT prefer longer, clearer, or more descriptive terms without first considering this rule.
- Secondary priority is on semantically correctness (e.g., "diabetes" → diagnosis terms, not cardiac terms)
- Common synonyms/abbreviations (e.g., "MI" ↔ "myocardial infarction")
- IMPORTANT: For `[NAME]` entities, map to the database entry with the highest lexical similarity (e.g., `"Leonado"` → `"Leonardo"`). You should never reject for name entities.

# Reject criteria (return index = -1):
- No lexical overlap AND semantically unrelated
- All scores very high (>20) with no common meaning
- Candidates are clearly wrong domain/category

CRITICAL: You CAN return best_match_index = -1 if no candidate is appropriate. Be selective.

# Tag derivation:
- Use table and column to create tag in the following format "TABLE.COLUMN".
- Make tags UPPERCASE with underscores, use dot to separate between table and column.

# Examples:

- If there are multiple exactly similar entity (by columns/tables) differing only by casing, choose the one with proper capitalization (e.g., "Penicilin" over "penicilin").
Entity: "penicillin"
Candidates:
  0. 'penicillin' (drugs, Score: 0)
  1. 'Penicillin' (drugs, Score: 0)
→ Select index 1, Tag: "DRUG" (proper capitalization)

Entity: "diabetes", Candidates: ['Diabetes Mellitus' (diagnoses.short_title), 'Diabetic' (procedures.short_title)]
→ Select index 0, Tag: "DIAGNOSIS.SHORT_TITLE" (highest overlapping + semantically correct)

Entity: "hypoxia", Candidates: ['hypoxia' (demographic.diagnoses), 'Hypoxemia' (diagnosis.short_title)]
→ Select index 0, Tag: "DEMOGRAPHIC.DIAGNOSIS" (highest overlapping + semantically correct)

Entity: "xyzabc", Candidates: ['Diabetes', 'Hypertension']
→ Select index -1, Tag: "NO_MATCH" (no similarity, likely invalid)

Entity: "cardiology", Candidates: ['Nephrology', 'Urology']
→ Select index -1, Tag: "NO_MATCH" (semantically unrelated)
"""

# ========== CBR-to-SQL: Template Construction ==========

sql_generation = """
Generate a SQL query by adapting retrieved examples to answer the original natural language question. Return "None" if insufficient information is provided to write a SQL query.

# Procedure
1. Select the most semantically aligned retrieved examples based on semantic intent.
2. Treat the selected examples as the authoritative template, and exactly follow its formulation.
3. Replace all entity placeholders (e.g., `[CONDITION]`, `[DRUG]`, `[DATE]`) using the provided entity mappings, or by guessing from retrieved examples and user queries.
4. Generate the final SQL query by reusing from selected similar example's logic, structure, and formatting. You have the option to return "None", if there is insuffient context to write a SQL query (impossible question).

# Rules

## 1. Example Selection Rule

1. Select the examples with the most aligned semantic meaning and logic. Then, Copy that examples' SQL queries' structure to adapt to the current questionc

2. There is always one filter clause for a condition mentioned in the original question. Do not attempt to address one condition value mutliple times.

## 2. IMPORTANT - Impossible Question Detection

Some questions are intentionally impossible to answer with the available data. Before writing SQL, verify that the question's logical form is fully supported by the retrieved examples and schema.
* Compare masked question forms (ignore entity values; focus on structure and intent) with the retrieved examples. If the retrieved examples are logically aligned in terms of logical form, then you can proceed. Otherwise, if there are any aspect of the questions that appears unanswerable, then return None.
* You must not rely on entity mappings to determine answerability. The presence of entity mappings does not guarantee that the question is answerable.

### Return `"None"` if ANY are true:
* Retrieved examples do **not** demonstrate how to answer **all semantic aspects** of the question's logical form
  *(Differences in entity values alone do NOT count as missing evidence.)*
* The schema lacks required tables or columns for **any** part of the question.
* You cannot confidently map **all question components** to database elements.

### Verification checklist (must pass all):
* Do retrieved examples share the **same logical form**, even with different entities?
* Does the schema contain **all required tables and columns**?
* Can **every condition and requirement** be expressed in SQL using the provided information?

### Important
* If retrieved examples share the same SQL structure and differ only by entity values, the question **IS answerable**.
* Do not rely on entity mappings to determine answerability. Always verify whether the question can be answered by checking the retrieved abstract, masked template as a logical form and the schema. 

## 3. Entity Replacement
Replace every placeholder with exactly one concrete value:
1. **Use entity mapping first**, consider the value proposed by the entity map.
2. **Verify semantic correctness:**
   - Does the mapped value match the entity's actual meaning?
   - Example: "dead" → EXPIRE_FLAG = 1 ✓ CORRECT | "dead" → "DAH" ✗ WRONG (unrelated)
3. **If entity mapping seems wrong or missing:**
   - Check if a different column better captures the intent (e.g., EXPIRE_FLAG for "dead")
   - Extract literal values directly from the question. Remember to utilize the full span of entities (as shown in the extracted values in the masked entity).
   - Consult retrieved examples for similar patterns
4. **Do NOT use fuzzy matches that are clearly semantically unrelated**
Critical: Correctness > Lexical similarity. Reject nonsensical mappings.

### 3. Output Constraints
* Return "None" if the input question is impossible to be answered.
* Otherwise, output ONLY one SQL query, no markdown, comments, explanations, or extra text. Closely the writing style of the retrieved SQLs (no linebreaks, etc.)
* Exactly one value per entity placeholder.
* Write SQL minimally, while strictly following the type, formatting, and logical conventions of the retrieved example.
"""
