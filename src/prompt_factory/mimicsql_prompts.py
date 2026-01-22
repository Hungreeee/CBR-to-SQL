# ========== RAG-to-SQL Prompts ==========

case_revising = """
Given past examples of SQL queries and a question, revise the examples to create a new SQL query that answers the question.

# Instructions:
- Sometimes, the SQL examples won't be sufficient to address all conditions in the query. Use what you know about the SQL schema to fulfill all conditions mentioned in the question.
- Make sure the new SQL query align directly with the question's intent.
- Follow exactly the SQL writing styles as in the given examples.
- If the retrieved examples are relevant to the user question, strictly follow the logic of the retrieved SQL examples.
- Your output should be only the SQL template, with no comments, explanations, or formatting.
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
- DRUG: Named medication (e.g., "metformin", "vancomycin")
- MEASUREMENT: Lab test names (e.g., "creatinine", "blood pressure")
- EQUIPMENT: Medical devices (e.g., "ventilator")
- ETHNICITY: Ethnic group (e.g., "Hispanic", "African American")
- RELIGION: Religion (e.g., "Christian Scientist", "Catholic")
- GENDER: Gender value (e.g., "Male", "Female")
- LANGUAGE: Language (e.g., "Spanish", "English")
- LOCATION: Specific places (e.g., "ICU", "Emergency Department")
- NAME: Human names (e.g., "John Smith")
- ADMISSION: Hospital admission (e.g., "Urgent")

STRUCTURAL ENTITIES (is_semantic = false):
Examples of values that are used as-is for filtering:
- ID: Patient identifiers (e.g., "12345", "SUBJ_001")
- ICD_CODE: Diagnosis codes (e.g., "29961", "V45.81", "250.00")
- AGE_VALUE: Age numbers (e.g., "65", "18")
- YEAR: Year values (e.g., "2019", "2150")
- NUMERIC_VALUE: Other numeric values (e.g., "2.0", "100")

# Extraction Rules:
1. IMPORTANT: **Do not extract generic words:**
   - List of generic keywords to IGNORE during EXTRACTION: "patient", "disease", "primary disease", "lab test", "procedure", "drug route", "route of drug administration", "hospital admission", "therapy", and similar terms.
      - "coronary artery primary disease" → extract "coronary artery", leaving "primary disease" because it is clearly similar to the generic terms.
      - "elective hospital admission" → extract "elective", leaving "hospital admission" because it is clearly similar to the generic terms.
2. **Extract COMPLETE medical phrases, but exclude generic terms**
   - Include everything: slashes (/), parentheses (), hyphens (-)
      - "aortic insufficiency/re-do sternotomy (aortic valve replacement)" → Extract ENTIRE phrase
      - "diabetes complicating a procedure" → Extract COMPLETE phrase
   - Exclude generic terms:
      - "primary disease called aortic insufficiency/re-do sternotomy (aortic valve replacement)" → extract everything BUT "primary disease", which is similar to the generic terms listed above.
3. **Preserve original form:**
   - Keep typos, capitalization, spacing, special characters exactly as written
   - "lipalse" → "lipalse" (not "lipase")
   - Extract only the specific value, leaving the generic terms behind (similar to the examples above).
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
- Evaluate candidates based on lexical similarity (highest priority) AND semantic meaning (second highest priority)
   - If a good match exists: Select the highest word overlap (considering also word ordering and special characters as needed) by choosing its index and derive tag from its table/column.
   - If NO good match exists: Set best_match_index = -1 and label = "NO_MATCH"
- The second highest priority is to verify semantic correctness:
   - Does the mapped value match the entity's actual meaning? If not, reject criteria (index = -1).
   - Example: "dead" → EXPIRE_FLAG = 1 ✓ CORRECT | "dead" → "DAH" ✗ WRONG (unrelated)
- Table/column preference can be used as a tie-breaker, but **must not override lexical priority**. However, if the question context hints a particular type of query, use it to guide selection among equally close matches. 
   IMPORTANT: 
   * Example: If the question contains keyword "primary disease", always prefer candidates from the DEMOGRAPHIC.DIAGNOSIS table/column over DIAGNOSIS._TITLE with similar matches. 
   * Example: If the question contains keyword "diagnosed with", prefer candidates from the DIAGNOSIS._TITLE table/column over DEMOGRAPHIC.DIAGNOSIS with similar matches (unless "primary disease" is mentioned in the same sentence).

# Accept criteria:
- Highest lexical match (similar form) OR substantial word overlap (considering also word ordering, extra special characters)
- Semantically correct (e.g., "diabetes" → diagnosis terms, not cardiac terms)
- Common synonyms/abbreviations (e.g., "MI" ↔ "myocardial infarction")
- For `[NAME]` entities, map to the database entry with the highest lexical similarity** (e.g., `"Leonado"` → `"Leonardo"`). You should never reject for name entities.

# Reject criteria (return index = -1):
- No lexical overlap AND semantically unrelated
- All scores very high (>20) with no common meaning
- Candidates are clearly wrong domain/category

CRITICAL: You CAN return best_match_index = -1 if no candidate is appropriate. Be selective.

# Tag derivation:
- Use table and column to create tag in the following format "TABLE.COLUMN".
- Make tags UPPERCASE with underscores, use dot to separate between table and column.

# Examples:
- IMPORTANT: If there are multiple exactly similar entity with similar columns/tables, differing only by casing, choose the one with better capitalization (e.g., "Penicilin" over "PENICILIN" over "penicilin") OR (e.g., "BusPIRone" over "Buspirone" over "buspirone" (more detailed casing))
Entity: "penicillin"
Candidates:
  0. 'penicillin' (drugs, Score: 0)
  1. 'Penicillin' (drugs, Score: 0)
  2. 'PENICILLIN' (drugs, Score: 0)
→ Select index 1, "Penicillin" (tag: DRUGS) because of better capitalization.

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
Generate a SQL query by adapting retrieved examples to address the original natural language question. 

# Procedure
1. Select the **most similar retrieved example(s)** based on natural language syntax (overlapping words, ordering of words, keywords, etc.) and semantic intent.
2. Treat the selected example as the **authoritative template**, and exactly follow its formulation, even though it may not make sense. 
3. Replace all entity placeholders (e.g., `[CONDITION]`, `[DRUG]`, `[DATE]`) using the provided entity mappings, or by guessing from retrieved examples and user queries.
4. Generate the final SQL query by **replicating the selected similar example's logic, structure, and formatting exactly**.

# Rules
## 1. Example Selection Rule
**Select the example with the highest word overlap with the original question:**
1. Count overlapping words/tokens between original question and each example question
2. Select the example with the most overlapping words (ignore semantic meaning, table types, or logic)
3. **Copy that example's SQL structure EXACTLY** - even if it seems wrong
**Critical: Token overlap > Everything else**
- If the most similar example has redundant clauses, unusual patterns, or logical errors → replicate them exactly
- If multiple examples conflict → choose the one with highest word overlap, then copy its SQL structure
4. There is always one filter clause for a condition mentioned in the original question. Do not attempt to address one condition value mutliple times.
5. For any question phrased as "which patients", "find patients", "get the list of patients", or similar variants, always interpret it as a request for the number of patients:
   Example: Which patients are medicated via the tp route?
   Correct: SELECT COUNT(DISTINCT patient_id) ...
   Incorrect: SELECT DISTINCT patient_id ...

Example: 
- Question: "Get the **list** of white-russian patients with neb route"
- Best match: "**Count** the number of white-russian patients who administer drugs via oral route"
- Copy: SELECT **COUNT** (DISTINCT ...) ← Copy the COUNT from example, NOT "list"

## 2. Entity Replacement
**Replace every placeholder with exactly one concrete value:**
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
* Output ONLY one SQL query, no markdown, comments, explanations, or extra text.
* Exactly one value per entity placeholder.
* Write SQL minimally, while strictly following the type, formatting, and logical conventions of the retrieved example.
"""
