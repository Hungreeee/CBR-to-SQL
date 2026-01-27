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
Before writing SQL, quickly assess if all the question's **core concepts** can be answered using the provided examples and schema.

### Quick Check:
1. **Identify the core concepts** - Remove all specific details. What general facts or relationships is the question asking for?
2. **Look for conceptual matches in examples** - Do the examples show queries about a similar *category* of information (e.g., counts, dates, events, assignments). You may interpret the meaning, but only in a tightly logical sense. 
3. **Verify schema mapping** - Can this concept be directly represented using the available columns?

### Mark as impossible only if:
- The **core concepts** (e.g., "consent status", etc.) are absent from all examples.
- The schema has **no columns** to directly represent that concept.

### Important:
- **Be literal with concepts, not details.** Match the *type* of query, not the exact scenario.
- **Examples provide proof of concept.** If examples show "maximum," then "minimum" is a valid concept. If they show "procedure occurred," that does not prove "consent was given" is stored. 
- **Avoid reinterpretation.** Do not soften wording or infer unstated relationships (e.g., "last ward ID" ≠ "ward ID that can admit").

**Examples:**
- ✅ *"What's the minimum daily prescription?"* → Concept: `aggregate metric (min)` of a `prescription`. Examples show `aggregate metric (sum)` of a `prescription`. **Answerable.** Saying it's possible would be a reasonable and logical interpretation.
- ❌ *"Has patient received a consent form?"* → Concept: `consent event`. Examples only show `procedure event`. **Impossible.** Saying it's possible would be over-interpretation.
- ❌ *"What ward can get patient in?"* → Concept: `ward suitability/eligibility`. Examples only show `historical ward assignment`. **Impossible.** Saying it's possible would be over-interpretation.

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
   - List of generic words to NOT INCLUDE in EXTRACTION: "patient", "disease", "primary disease", "lab test", "procedure", "drug", "drug route", "route of drug administration", "hospital admission", "therapy", "prescription", "fluid", and similar terms. Instead, prioritize extracting specific details, for example, the NAME of the diagnosis/procedures/drugs/patients/etc., any numerical values, and the NAMED types of related items.
      - "coronary artery primary disease" → extract "coronary artery" the NAME, leaving "primary disease" because it is clearly similar to the generic terms. 
      - "elective hospital admission" → extract "elective" the NAMED type, leaving "hospital admission" because it is clearly similar to the generic terms. 
      - "peura fluid" → extract "peura" the NAME, leaving "fluid" because it is within the list of generic terms.
2. **Extract COMPLETE medical phrase NAMES, but exclude generic terms**
   - Include everything: slashes (/), parentheses (), hyphens (-)
      - "aortic insufficiency/re-do sternotomy (aortic valve replacement)" → Extract ENTIRE phrase
      - "diabetes complicating a procedure" → Extract COMPLETE phrase
   - Exclude generic terms:
      - "primary disease called aortic insufficiency/re-do sternotomy (aortic valve replacement)" → extract everything BUT "primary disease", which is similar to the generic terms listed above.
3. **Preserve original form:**
   - Keep typos, capitalization, spacing, special characters exactly as written
   - "lipalse" → "lipalse" (not "lipase")
   - Extract only the specific value, leaving the generic terms behind (similar to the examples above).
4. **Compound Entity Extraction**
   - IMPORTANT: Extract NAMED compound entities as a SINGLE span when connected by conjunctions (and/or/but/not) or punctuation (/, ;, comma).
   - Do NOT split into separate entities - capture the entire phrase.
   Examples:
   - "diabetes and sepsis" → Extract: "diabetes and sepsis" ✓
   - "aspirin/ibuprofen" → Extract: "aspirin/ibuprofen" ✓
   Do NOT extract as separate entities:
   - ✗ "diabetes", "sepsis"
   - ✗ "aspirin", "ibuprofen"
5. **Tagging decision:**
   - Medical/demographic term that might have variations → is_semantic = true
   - ID/number/code for exact matching → is_semantic = false
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

## 1. Example Selection Rule
1. Select the examples with the most aligned semantic meaning and logic. Then, Copy that examples' SQL queries' structure to adapt to the current questionc
2. There is always one filter clause for a condition mentioned in the original question. Do not attempt to address one condition value mutliple times.

## 2. Question Answerability Check
Before writing SQL, quickly assess if all the question's **core concepts** can be answered using the provided examples and schema.

### Quick Check:
1. **Identify the core concepts** - Remove all specific details. What general facts or relationships is the question asking for?
2. **Look for conceptual matches in examples** - Do the examples show queries about a similar *category* of information (e.g., counts, dates, events, assignments). You may interpret the meaning, but only in a tightly logical sense. 
3. **Verify schema mapping** - Can this concept be directly represented using the available columns?

### Mark as impossible only if:
- The **core concepts** (e.g., "consent status", etc.) are absent from all examples.
- The schema has **no columns** to directly represent that concept.

### Important:
- **Be literal with concepts, not details.** Match the *type* of query, not the exact scenario.
- **Examples provide proof of concept.** If examples show "maximum," then "minimum" is a valid concept. If they show "procedure occurred," that does not prove "consent was given" is stored. 
- **Avoid reinterpretation.** Do not soften wording or infer unstated relationships (e.g., "last ward ID" ≠ "ward ID that can admit").

**Examples:**
- ✅ *"What's the minimum daily prescription?"* → Concept: `aggregate metric (min)` of a `prescription`. Examples show `aggregate metric (sum)` of a `prescription`. **Answerable.** Saying it's possible would be a reasonable and logical interpretation.
- ❌ *"Has patient received a consent form?"* → Concept: `consent event`. Examples only show `procedure event`. **Impossible.** Saying it's possible would be over-interpretation.
- ❌ *"What ward can get patient in?"* → Concept: `ward suitability/eligibility`. Examples only show `historical ward assignment`. **Impossible.** Saying it's possible would be over-interpretation.

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

### 4. Output Constraints
* Return "None" if the input question is impossible to be answered.
* Otherwise, output ONLY one SQL query, no markdown, comments, explanations, or extra text. Closely the writing style of the retrieved SQLs (no linebreaks, etc.)
* Exactly one value per entity placeholder.
* Write SQL minimally, while strictly following the type, formatting, and logical conventions of the retrieved example.
"""
