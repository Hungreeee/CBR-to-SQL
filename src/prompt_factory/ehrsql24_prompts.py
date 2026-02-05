# ========== RAG-to-SQL Prompts ==========

case_revising = """
Generate a SQL query by adapting retrieved examples to answer the original natural language question. Return "None" if insufficient information is provided to write a SQL query.

# Procedure
1. Select the most semantically aligned retrieved examples based on semantic intent.
2. Treat the selected examples as the authoritative template, and exactly follow its formulation.
3. Generate the final SQL query by reusing from selected similar example's logic, structure, and formatting. You have the option to return "None", if there is insuffient context to write a SQL query (impossible question).

## 1. Example Selection Rule
1. Select the examples with the most aligned semantic meaning and logic. Then, Copy that examples' SQL queries' structure to adapt to the current questionc
2. There is always one filter clause for a condition mentioned in the original question. Do not attempt to address one condition value mutliple times.

## 2. Question Answerability Check
Before writing SQL, quickly assess if all the question's core concepts can be answered using the provided examples and schema.

### Quick Check:
1. Identify the core concepts - Remove all specific details. What general facts or relationships is the question asking for?
2. Look for conceptual matches in examples - Do the examples show queries about a similar *category* of information (e.g., counts, dates, events, assignments). You may interpret the meaning, but only in a tightly logical sense. 
3. Verify schema mapping - Can this concept be directly represented using the available columns?

### Mark as impossible only if:
- The core concepts (e.g., "consent status", etc.) are absent from all examples.
- The schema has no columns to directly represent that concept.

### Important:
- Be literal with concepts, not details. Match the *type* of query, not the exact scenario.
- Examples provide proof of concept. If examples show "maximum," then "minimum" is a valid concept. If they show "procedure occurred," that does not prove "consent was given" is stored. 
- Avoid reinterpretation. Do not soften wording or infer unstated relationships (e.g., "last ward ID" ≠ "ward ID that can admit").

Examples:
- ✅ *"What's the minimum daily prescription?"* → Concept: `aggregate metric (min)` of a `prescription`. Examples show `aggregate metric (sum)` of a `prescription`. Answerable. Saying it's possible would be a reasonable and logical interpretation.
- ❌ *"Has patient received a consent form?"* → Concept: `consent event`. Examples only show `procedure event`. Impossible. Saying it's possible would be over-interpretation.
- ❌ *"What ward can get patient in?"* → Concept: `ward suitability/eligibility`. Examples only show `historical ward assignment`. Impossible. Saying it's possible would be over-interpretation.

## 3. Output Constraints
* Return "None" ONLY IF the input question is impossible to be answered.
* Otherwise, output ONLY one SQL query, no markdown, comments, explanations, or extra text. Closely the writing style of the retrieved SQLs (no linebreaks, etc.)
* Exactly one value per entity placeholder.
* Write SQL minimally, while strictly following the type, formatting, and logical conventions of the retrieved example.
"""
 
# ========== CBR-to-SQL: Source Discovery - Iterative Refinement ==========

entity_extraction = """
Your task is to extract text spans that correspond to standardized database concepts (e.g., diseases, procedure names, diagnostic labels, drugs, patient names, or coded attributes, etc.). This is a text alignment task, by extracting the named entities, you help to find out the real database-specific values of these entities, while masking specific information away, leaving behind the logical form template of the question. Therefore, you must leave enough generic words behind so the sentence is still understandable without the extracted specific information.

For each extracted entity, you must:
1. Extract the complete value span (preserve typos, special characters, parentheses, slashes) while ignoring generic terms (given as a list below).
2. Assign a temporary tag based on sentence context.
3. Flag whether it's semantic (needs database lookup) or structural (use as-is).

# Example Entity Types:

SEMANTIC ENTITIES (is_semantic = true):
Examples of entities that need database lookup validation:
- CONDITION: Specific diagnosis, disease (e.g., "diabetes", "sepsis")
- PROCEDURE: Specific intervention or test (e.g., "MRI scan", "cardiac catheterization")
- DRUG: Named medication (e.g., "stool", "sonata", "impact")
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
- AGE_VALUE: Age numbers (e.g., "65",w "18")
- NUMERIC_VALUE: Other numeric values (e.g., "2.0", "100")

# Extraction Rules:
1. IMPORTANT: Do not extract generic words:
   - List of generic words to NOT INCLUDE in EXTRACTION: "microbiology test", "microbiological", "patient", "intake", "lab test", "procedure", "drug", "drug route", "hospital admission", "therapy", "prescription", "hospital careunit", and similar terms. Instead, prioritize extracting specific details, for example, the NAME of the diagnosis/procedures/drugs/patients/etc., any numerical values, and the NAMED types of related items.
      - "postmortem culture microbiology test" → extract "postmortem culture" the NAME, leaving "microbiology test" because it is clearly similar to the generic terms. 
      - "true urine output" → extract "true urine" the NAMED type, leaving "output" because it is clearly similar to the generic terms. 
2. Extract COMPLETE medical phrase NAMES, but exclude generic terms
   - Include all special charactes if it belongs to the entity span: slashes (/), parentheses (), hyphens (-), etc.
      - "po/ng" → Extract ENTIRE phrase "po/ng"
      - "drainage of pericardial cavity with drainage device, percutaneous" → Extract COMPLETE phrase
      - "Count the patients who received control of epistaxis by anterior nasal packing previously during the same hospital encounter?"
      -> Entity is "control of epistaxis by anterior nasal packing".
      You must remember that you are hiding specific NAMES away, not generic words. 
   - Exclude generic terms:
      - "primary disease called aortic insufficiency/re-do sternotomy (aortic valve replacement)" → extract everything BUT "primary disease", which is similar to the generic terms listed above.
      BAD EXAMPLE: Could you show me the top four most common diagnoses?
      While it may seem like "top four most common diagnoses" is an entity, it is obviously not. Because without this, the question lost all of its structural semantics. 
      Remember, you are to leave the logical form structure behind, extracting only specific, NAMED, numerical, etc. information! 
3. Preserve original form:
   - Keep typos, capitalization, spacing, special characters exactly as written
   - "lipalse" → "lipalse" (not "lipase")
   - Extract only the specific value, leaving the generic terms behind (similar to the examples above).
4. Complete Entity Extraction:
   - IMPORTANT: Extract NAMED compound entities as a SINGLE span when connected by punctuation (/, ;, comma). Do NOT split into separate entities - capture the entire phrase.
   Examples:
   - "*nf* nicardipine hcl iv" → Extract: "*nf* nicardipine hcl iv" ✓
   - "aspirin/ibuprofen" → Extract: "aspirin/ibuprofen" ✓
5. Tagging decision:
   - Medical/demographic term that might have variations → is_semantic = true
   - ID/number/code for exact matching → is_semantic = false
"""

tag_assignment = """
You are given a an entity value. Your task is to select one highest matching value from the list of real database entities OR reject all if no good match exists.

# Instructions:
Choose its index and derive the label from its table/column. The following rules must be applied:
- Evaluate candidates using surface-level lexical similarity (character + word overlapping) as the highest priority, and semantic correctness as a secondary priority.
   - Select the highest word/character overlap (considering also word ordering and special characters as needed) by choosing its index and derive tag from its table/column.
   - IMPORTANT: If a shorter or abbreviated term has the highest overlapping, it must be selected, even if a longer term appears semantically clearer. Do NOT prefer longer, clearer, or more descriptive terms. 
- The second highest priority is to verify semantic correctness:
   - Does the mapped value match the entity's actual meaning? If not, reject criteria (index = -1).
   - Example: "dead" → EXPIRE_FLAG = 1 ✓ CORRECT | "dead" → "DAH" ✗ WRONG (unrelated)
   - This rule should be excluded if the entity has an exact character-by-character match. As mentioned, always prefer word/character overlapping as highest priority matching.
- Table/column preference can be used as a tie-breaker, but must not override lexical priority. However, if the question context specifies a particular type of query, use it to guide selection among equally close matches. 
- If the same entity appear accross multiple tables/columns, use the question semantics to determine the type of entity it may be. 
   Example: How many patients have previously received hypothermia within 2 months?
   "hypothermia" - DIAGNOSIS.SHORT_TITLE
   "hypothermia" - PROCEDURES.SHORT_TITLE
   While both are valid concepts on their own, the question hints that the value should be a procedure (you can receive a procedure, not a diagnosis).

# Accept criteria:
- IMPORTANT: Always prefer the term with the highest word-level or character-level similarity, or with substantial word overlap (including word order and special-character variations). Do NOT prefer longer, clearer, or more descriptive terms without first considering this rule.
- Secondary priority is on semantically correctness (e.g., "diabetes" → diagnosis terms, not cardiac terms).
- Common synonyms/abbreviations (e.g., "MI" ↔ "myocardial infarction")
- IMPORTANT: For `[NAME]` entities, map to the database entry with the highest lexical similarity (e.g., `"Leonado"` → `"Leonardo"`). You should never reject for name entities.

# Reject criteria (return index = -1):
- If NO good match exists: Set best_match_index = -1 and label = "NO_MATCH"
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


sql_generation = """
**SQL Generation with Mandatory Validation Protocol**

Generate a SQL query by adapting retrieved examples to directly answer the original question. Return **"None"** if the question cannot be answered with available information.

## **Two-Phase Validation Protocol**

**PHASE 1: REQUIRED INFORMATION CHECK**
Before any analysis, verify the question provides all necessary information:
- Extract every required piece of information from the question
- Check that each piece is **concretely specified** (no placeholders, no too vague references (e.g., "this date" is acceptable because it still points to a relative date, but "that date" is not (which one exactly))).
- **IMMEDIATE FAILURE**: If any specific information is missing to answer the query → Return "None"

**PHASE 2: DATABASE FEASIBILITY CHECK**
Validate that the database can store and retrieve the requested information:
- For each required information element, identify the **exact database column** that stores it
- **CRITICAL RULE**: The column must **directly store** that exact type of information. You may infer this from the schema and the examples. However, you must be strict when comparing concepts, because the question will sometimes introduce similar concepts to trick you.
   EXAMPLE: 
      - "specimen quality" (impossible, no column) vs "specimen type" (possible, "spec_type_desc"). Such concepts are not limited to only this example, so be strict while verifying the similarity of such concepts.
      - "hospital visits/arrivals" (impossible) vs "hospital admissions" (possible, addmission)
- **IMMEDIATE FAILURE**: If any element has no direct column match or missing information → Return "None".
   EXAMPLE: Get the patient information in this specific date (which information, which specific date?) -> immediately return "None"

**ONLY IF ALL TWO PHASES PASS**: Proceed to SQL generation

## **SQL Generation Process**
1. **Classify Question Intent**:
   - **Existence** ("Has there been...") → `SELECT COUNT(*) > 0`
   - **Temporal** ("When did...") → Return timestamp
   - **Aggregation** ("How many...") → Use appropriate function (return raw values, no extra filters or rounding, exactly what the question asked for).
   EXAMPLE: How many hours has passed since ... -> give number of hours as decimals, no abitrary rounding.
   HOURS = strftime('%s', 'now') - strftime('%s', prescriptions.starttime)) / 3600.0
   - **Fact** ("What is...") → Return value without aggregation unless specified

2. **Adapt Examples as Appropriate**:
   - Use examples to adapt similar SQL structure. You may read the example questions and the original questions to verify the similarities, then proceed to adapt the corresponding SQL.
   - Note that the original question is authorative, and the examples are only support. In certain cases, the examples may not be able to provide enough help, so you must write the SQL on your own rather then enforcing the example templates to the output SQL. Always prioritize writing SQL that directly answers the original question first.
   - IMPORTANT: Replace placeholders with concrete values from entity mappings.
   - Include only necessary clauses that directly serve the question. Make sure that all aspects of the questions are appropriately covered.
   - The formatting style of the SQL query must be similar to the examples (only the small formatting tips like the having no linebreaks, capitalization style, not follow logic strictly).

3. **Time Expression Handling**: Be extremely careful with time expressions:
     - `this year/mm` or `this month/dd` → filter by corresponding format. Never assume the exact year, month, or date for relative time references (this month, last year, etc.), so always use relative filtering for relative references, exact filtering for exact time references. 
      Example: Assuming "this month" is "01" is WRONG.
     - `X years ago` → timestamp = datetime(reference_time, '-X years') (exact reference point as rolling window, do not truncate to year)
     - `until/before X years ago` → timestamp <= datetime(reference_time, '-X years')
     - `since X years ago` → timestamp >= datetime(reference_time, '-X years') (as datetime, not just year)
     - `during year X` → timestamps in year X (within calendar year, not rolling window)
     - `current visit` → timestamp is null (visit hasn't been completed)
     - `first/last visit` → timestamp is not null (visit has been completed)
     - **Always filter years with `%Y`**, never `%y`

## **Critical Rules for Answerability**

**UNANSWERABLE QUESTIONS (Return "None"):**
1. Questions with **missing concrete information** (too vague references, placeholders)
2. Questions requesting **information not stored in database** (no corresponding column)
3. Questions requiring **operations not demonstrated in examples**
4. Questions with **semantically impossible mappings** (diagnosis → test column)

**ANSWERABLE QUESTIONS (Generate SQL):**
1. All required information is concretely specified
2. Each information element has a direct database column
3. Examples demonstrate similar query patterns

## **Output Specification**
- **Unanswerable**: Return exactly "None" (no SQL, no explanation)
- **Answerable**: Return exactly one SQL query (no additional text, no markdown)
- **Formatting**: Match example style exactly (no arbitrary rounding, no linebreaks)

**Final Decision Flow:**
```
1. Does question provide all concrete information? → No → "None"
2. Does database have columns for all information? → No → "None"  
3. All checks pass → Generate SQL from examples
```
"""

# ========== CBR-to-SQL: Template Construction ==========

# sql_generation = """
# Generate a SQL query by adapting retrieved examples to directly answer the original natural language question. Return "None" if insufficient information is provided to write a SQL query.

# ## **Procedure**
# 1. Select the retrieved example(s) with the closest semantic intent.
# 2. Treat selected examples only as **templates** to adapt and the original question as **authoritative**. Replicate the chosen example's **logic, structure, and formatting exactly**, adapting only what is necessary. 
# 3. Replace all entity placeholders (e.g., `[DATE]`, `[DRUG]`) using entity mappings, or infer from examples and the question.
# 4. IMPORTANT: Return the string "None" immediately if there is insufficient context to support a valid SQL query.

# ## 1. Intent Analysis & SQL Writing Rules:
# * **Step 1: Determine the exact query intent** - Analyze what the question is fundamentally asking for:
#    - **Fact retrieval**: Asking for specific facts, measurements, or values
#    - **Existence check**: Asking whether something exists/has occurred (yes/no questions)
#    - **Temporal query**: Asking for time-related information (when, how long, timing)
#    - **Aggregation**: Asking for counts, averages, maximums, minimums
#    - **Comparison**: Asking for relationships between values
   
#    **CRITICAL**: The question's wording determines what you should return in the SELECT clause and overall query structure. The temporal aspects mentioned in examples may only be conditional filters, not the actual answer being requested.

# * **Step 2: Match SQL structure to intent**:
#    - **Existence questions** ("Has there been...", "Is there...", "Does...exist") → Use `SELECT COUNT(*) > 0`
#    - **Temporal questions** ("When did...", "What time...") → Return timestamps
#    **Fact questions** ("What is...", "What was...") → Return specific values.
#       - You may use aggregation functions (MAX, MIN, SUM, AVG) only when implied by the question's context. Apply the minimal aggregation needed:
#          - "What is the **total** output..." → Use `SUM()`
#          - "What is the **maximum** value..." → Use `MAX()`
#          - "What is the **average**..." → Use `AVG()`
#       - "What is the **first/last** measurement..." → Use ordering with `LIMIT 1` in appropriate direction
#       - **CRITICAL**: When the question asks for a singular fact without implying aggregation (e.g., "What is the patient's blood pressure?"), return the specific value without aggregation unless contextual clues clearly indicate otherwise. The presence of temporal conditions does not automatically justify aggregation.
#    - **Count questions** ("How many...") → Use `COUNT()` aggregation

# * **Step 3: Adapt examples carefully**:
#    - Ensure **every condition in the question maps to exactly one SQL constraint**
#    - **Do not copy irrelevant clauses** from examples - each clause must directly serve the question's intent
#    - **Verify column usage**: Match entity mappings to correct database columns; don't assume examples use the right columns
#    - The SELECT clause must return **exactly what the question asks for**, no more, no less
#    - Follow closely the style and formatting of the examples AND the intent of the question. Do not abitrarily apply functions/formats without specifications (e.g., abitrary rounding, linebreaks).

# * **IMPORTANT: Time filtering precision**:
#   - Distinguish between **time conditions** (filters in WHERE clause) and **time answers** (values in SELECT clause)
#   - When filtering by time, ensure the filter granularity matches the question's time reference
#   - For relative time references, use appropriate datetime calculations
#   - Time conditions should constrain the dataset; time answers should provide requested temporal information

# ## 2. Adaptation Guidelines:
# 1. **Question-first adaptation**: Start from what the question needs, then see which example structures can support it
# 2. **Column validation**: Cross-reference entity mappings with example queries to ensure correct column usage
# 3. IMPORTANT: **Intent alignment**: If an example's SELECT clause doesn't match the question's intent, adapt it or find a better example
# 4. **Minimal implementation**: Include only the SQL necessary to answer the question directly and completely
# 5. IMPORTANT: **Style consistency**: Maintain the database's SQL dialect and formatting style from examples (no abitrary rounding up, no linebreaks, etc.)

# ## 3. Question Answerability Check
# IMPORTANT: Before writing SQL, quickly assess if the question can be answered using available information.

# ### Quick Check Process:
# 1. **Identify core query pattern** - What is the fundamental structure being asked? Look for:
#    - Main entities being queried (tests, diagnoses, medications, etc.)
#    - Relationships between entities (after diagnosis, same encounter, etc.)
#    - Aggregations needed (top N, counts, frequencies)
# 2. **Look for structural matches in examples**:
#    - Do examples show similar **query patterns** (e.g., "top N most frequent X after Y")?
#    - Are the **required joins and conditions** demonstrated in examples?
#    - Is the **overall query logic** similar, even if details differ?
# 3. **Schema and entity mapping validation**:
#    - Check if entity mappings provide reasonable column matches
#    - If mappings seem incorrect but examples show the correct pattern, trust the examples
#    - Examples demonstrating complex queries with specific conditions are strong evidence of answerability

# ### Answerability Decision Rules:
# ✅ **Answerable if**:
# - Examples show queries with the same **structural pattern** (joins, conditions, aggregations)
# - The **database schema supports** the required relationships (joins between tables exist)
# - Entity mappings provide **plausible column matches**, even if not perfect
# - **Multiple examples** demonstrate similar complex query patterns

# ❌ **Return "None" only if**:
# - Core concepts are **completely absent** from schema and examples (e.g., "consent forms" when only procedures exist)
# - Required **relationships cannot be expressed** with available schema
# - **No examples** show even remotely similar query patterns
# - Entity mappings are **fundamentally impossible** (e.g., mapping diagnosis names to test types)

# ### Critical Guidance:
# - **Examples provide proof of structural feasibility**. If examples show complex diagnosis+test queries, similar questions are likely answerable.
# - **Entity mappings are suggestions, not constraints**. If examples use different columns for similar concepts, follow the examples.
# - **Focus on query structure, not exact wording**. "Specimens tested" ≈ "microbiology tests" in examples.
# - **Diagnosis names** typically map to `d_icd_diagnoses.long_title` and `diagnoses_icd.icd_code`, not to microbiology columns.

# **When in doubt, write the SQL query** using the closest example pattern. Only return "None" for clearly impossible questions where examples provide no structural guidance.

# Examples:
# - ✅ *"What's the minimum daily prescription?"* → Concept: `aggregate metric (min)` of a `prescription`. Examples show `aggregate metric (sum)` of a `prescription`. Answerable. Saying it's possible would be a reasonable and logical interpretation.
# - ❌ *"Has patient received a consent form?"* → Concept: `consent event`. Examples only show `procedure event`. Impossible. Saying it's possible would be over-interpretation.
# - ❌ *"What ward can get patient in?"* → Concept: `ward suitability/eligibility`. Examples only show `historical ward assignment`. Impossible. Saying it's possible would be over-interpretation.

# ## 4. Entity Replacement
# Replace every placeholder with exactly one concrete value:
# 1. Use entity mapping first, consider the value proposed by the entity map.
# 2. Verify semantic correctness:
#    - Does the mapped value match the entity's actual meaning?
#    - Example: "dead" → EXPIRE_FLAG = 1 ✓ CORRECT | "dead" → "DAH" ✗ WRONG (unrelated)
# 3. If entity mapping seems wrong or missing:
#    - Check if a different column better captures the intent (e.g., EXPIRE_FLAG for "dead")
#    - Extract literal values directly from the question. Remember to utilize the full span of entities (as shown in the extracted values in the masked entity).
#    - Consult retrieved examples for similar patterns
# 4. Do NOT use fuzzy matches that are clearly semantically unrelated
# Critical: Correctness > Lexical similarity. Reject nonsensical mappings.

# ### 5. Output Constraints
# * Return "None" if the input question is impossible to be answered.
# * Otherwise, output ONLY one SQL query, no markdown, comments, explanations, or extra text. IMPORTANT: Closely the writing style of the retrieved SQLs (no linebreaks, abitrary rounding, etc.)
# * Exactly one value per entity placeholder.
# * Write SQL minimally, while strictly following the formatting conventions of the retrieved example.
# """

prompt_decomposition = """
Your task is to decompose a complex natural language question into 2-3 simpler sub-questions that can help retrieve relevant SQL examples from a case base.

## **Objective**
Break down the original question into independent sub-questions that capture different aspects or components of the query. Each sub-question should represent a retrievable concept that might appear in example cases.

## **Decomposition Rules**
1. **Identify core components**: Break the question into logical parts (entities, conditions, operations, temporal constraints).
2. **Create independent sub-questions**: Each sub-question should stand alone and be answerable independently.
3. **Preserve key concepts**: Maintain the specific entities, temporal references, and operations from the original question.
4. **Aim for 2-3 sub-questions**: Too many fragments lose context; too few miss retrieval opportunities.
5. **Do NOT rewrite or simplify entities**: Keep medical terms, codes, and specific values exactly as written.

## **What to Extract as Sub-Questions**
- **Entity-focused**: Questions about specific conditions, procedures, drugs, or measurements
- **Temporal constraints**: Questions about time windows, sequences, or date filters
- **Aggregations**: Questions about counts, averages, maximums, or other summary statistics
- **Relationships**: Questions about patient-event relationships or multi-step logic

## **What NOT to Extract**
- Do not create sub-questions that are too generic (e.g., "What is a patient?")
- Do not split compound entities connected by conjunctions (e.g., "diabetes and hypertension" stays together)
- Do not add information not present in the original question

## **Examples**

**Original Question**: "How many patients diagnosed with sepsis in 2019 received vasopressors within 24 hours?"

**Sub-questions**:
1. How many patients were diagnosed with sepsis in 2019?
2. Which patients received vasopressors within 24 hours?
3. How to filter patients by diagnosis and drug administration timing?

---

**Original Question**: "What was patient 12345's blood pressure the first time they visited the ICU?"

**Sub-questions**:
1. What was the blood pressure measurement for patient 12345?
2. When was patient 12345's first visit to the ICU?

---

**Original Question**: "List all procedures performed on patients with diabetes who expired during their hospital stay."

**Sub-questions**:
1. Which patients have diabetes?
2. Which patients expired during their hospital stay?
3. What procedures were performed on specific patients?


**Original Question**: "What is the average age of patients who had an MRI scan last year?"

**Sub-questions**:
1. What is the average age of patients?
2. Which patients had an MRI scan last year?

## **Output Format**
Return ONLY a list of sub-questions. No explanations, no markdown, no extra text.
"""