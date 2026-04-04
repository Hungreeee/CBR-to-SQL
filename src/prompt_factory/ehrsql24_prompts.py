# ========== RAG-to-SQL Prompts ==========

case_revising = """
# SQL Query Generation from Natural Language

Generate a SQL query by adapting retrieved examples to answer the natural language question. Return **"None"** if the question cannot be answered with available information.

## Core Task Requirements

This is a high-stakes evaluation requiring exact, minimal responses. Your SQL must directly and completely answer the question using only information available in the schema and examples. The question is authoritative; examples serve only as structural templates.

## Two-Phase Validation Protocol

Execute these phases sequentially. Failure at any phase requires immediate return of "None".

### Phase 1: Question Completeness Verification

Verify the question provides all concrete information needed:

- Extract every entity, value, and condition the question references
- Confirm each is explicitly specified without ambiguity or placeholders
- Relative references like "this month" or "last year" are acceptable
- Vague references like "that date", "those patients", or "the procedure" without context are not acceptable

**Return "None" immediately if**: Any specific information needed to answer is missing or too vague to determine.

### Phase 2: Database Feasibility Verification

For each entity or concept in the question, perform exact column matching:

1. **Identify the exact database column** that stores this information
2. **Verify semantic precision**: The column must store the exact concept requested, not a similar or related concept
3. **Document the match**: Column name and table must be cited from schema or examples

**Critical Matching Rules**:

- **Exact attribute matching required**: If the question asks for "quality", a column storing "type" is not acceptable even if both relate to the same entity. Example: "specimen quality" cannot map to a column storing "specimen type" - these are different attributes requiring different columns.

- **Prefer specific over generic labels**: When multiple columns exist, use the most complete and specific one. Example: prefer "height (cm)" over an abbreviation "height" when both exist, as the former is more precise.

- **Table-specific columns**: Some concepts exist in multiple tables with different meanings. Lab tests like "oxygen" or "hemoglobin" use labevents table; vital signs like "temperature" or "blood pressure" use chartevents table; inputs use inputevents table; outputs use outputevents table. Verify you're using the correct table for the concept.

- **Distinguish similar concepts strictly**: The question may introduce subtly different concepts to test precision. You must be extremely strict while judging the feasibility of a concept (i.e., can be mapped to a specific column in the schema). Examples of distinct concepts that are similar, but cannot be mapped to each other and is essentially different:
  - "Specimen quality" (impossible) vs. "specimen name/type" (possible) (different attributes of same entity)
  - "hospital visits" (impossible) vs. "hospital admission" (possible) (different events)
  - "Performing physician" (impossible) vs. "patient" (possible) (different entities)
  - Temperature vs. temperature celsius (different measurement units/standards)

**Return "None" immediately if**:
- Any entity or concept has no exact matching column
- Semantic mismatch exists (similar but not identical concepts)
- Required relationship or join path doesn't exist in schema
- Column stores related but different information than requested

**Only proceed to SQL generation if both phases pass completely.**

## SQL Generation Process

### Step 1: Classify Query Intent

Determine what type of answer the question expects:

- **Existence queries** ("Has there been...", "Is there...", "Does...exist"): Return `SELECT COUNT(*) > 0`
- **Temporal queries** ("When did...", "What time..."): Return timestamp values
- **Count queries** ("How many..."): Return `COUNT()` aggregation
- **Fact queries** ("What is...", "What was..."): Return specific values with careful consideration of aggregation:
  - Use aggregation (SUM, MAX, MIN, AVG) only when explicitly stated or clearly implied by context
  - Singular form with temporal range often implies aggregation: "What was the total output during March" requires SUM
  - Without clear aggregation signal, return individual values or use ordering with LIMIT
  - "What was the dosage prescribed" with multiple prescriptions requires clarification from context: if asking for total, use SUM; if asking for a specific instance, use temporal ordering with LIMIT

### Step 2: Adapt Example Structure

Use retrieved examples as structural templates:

- Examine example questions to find similar query patterns
- Replicate the SQL structure, formatting style, and join patterns from examples
- The original question is authoritative - if examples don't perfectly fit, write SQL that directly answers the question rather than forcing an example template

**Structure Adaptation Rules**:
- Include only clauses that directly serve the question's requirements
- Every condition in the question must map to exactly one SQL constraint
- Do not copy irrelevant filters or clauses from examples
- Ensure all aspects of the question are covered in the SQL

### Step 3: Time Expression Handling

Time references require precise interpretation based on patterns demonstrated in the retrieved examples.

**Core principle:** If examples demonstrate datetime functions (`datetime()`, `strftime()`), then time-based filtering operations are supported. Study examples to learn the specific SQL patterns this database uses.

**Relative Time References** (computed dynamically from reference point):

Questions using relative time language require dynamic computation:
- **Period boundaries** (this year, this month, last year): Filter by normalizing timestamps to period starts. Examples will show functions like `datetime(time, 'start of year')` or similar period boundary operations.

- **Combined period + day** (this month/15, last year/March): Apply period boundary filter AND day/month component filter. Examples will show both boundary functions and component extraction functions like `strftime('%d', time)`.

- **Rolling windows** (X days ago, since X months ago): Calculate exact time offsets from reference point - not calendar boundaries. Examples will show offset syntax like `datetime(reference, '-X unit')` where unit is year/month/day.

- **General pattern**: Relative expressions use datetime arithmetic with reference points. Never hardcode relative references to absolute values.

**Absolute Time References** (direct value comparison):

Questions using specific dates/years compare against fixed values:
- **Year/month/date filters**: Extract time components and compare to specific values. Examples will show functions like `strftime('%Y', time) = '2100'` or similar component extraction patterns.

- **General pattern**: Absolute expressions extract components and compare to literals.

**Visit/Encounter Status Indicators**:

Questions about "current" vs "completed" visits use status fields:
- **Ongoing status**: Identified by NULL end/discharge timestamps
- **Completed status**: Identified by NOT NULL end/discharge timestamps
- Examples will show NULL checks on discharge/end time fields

**Duration Calculations**:

When computing time elapsed between two timestamps:
- Convert timestamps to common unit (seconds, days, etc.) using timestamp arithmetic
- Perform subtraction in that unit
- Convert to requested output unit if needed
- Examples will show specific conversion formulas - follow those patterns exactly
- Preserve decimal precision unless examples demonstrate rounding

### Step 4: Aggregation Decision Logic

Apply aggregation functions carefully based on question semantics:

**Apply aggregation when**:
- Explicitly stated: "total", "sum", "average", "maximum", "minimum", "count"
- Singular form implying combination: "What was the dosage prescribed" when multiple prescriptions exist typically means total dosage
- Temporal range with singular measurement: "the output during March" implies total output across the month

**Do not apply aggregation when**:
- Question asks for specific instance: "first", "last", "most recent" → use ORDER BY with LIMIT
- Question asks for individual values: "What are the measurements" (plural)
- Question provides sufficient constraints to return single record without aggregation

**When uncertain**: Examine similar patterns in provided examples for guidance.

## Example-Based Learning

Study the provided examples to understand:
- SQL dialect and formatting conventions (capitalization, spacing, no linebreaks)
- Join patterns for relating tables
- Subquery structures for complex filtering
- Time filtering techniques
- No arbitrary transformations like ROUND() unless demonstrated in examples

## Output Requirements

**Format**:
- If unanswerable: Return exactly the string "None" with no additional text
- If answerable: Return exactly one SQL query with no markdown formatting, no comments, no explanations
- Follow the exact formatting style of examples: no linebreaks within queries, consistent capitalization, spacing patterns

**Style Consistency**:
- Match example SQL formatting precisely
- Do not add arbitrary functions (ROUND, CAST) unless shown in examples
- Do not modify datetime formats unless examples demonstrate the pattern
- Use same quotation style, comma placement, and indentation as examples

## Decision Flowchart

Before generating any SQL, complete this checklist:

1. Does the question provide all concrete information needed? → If No: Return "None"
2. Can each entity/concept map to an exact database column? → If No: Return "None"
3. Are all concepts semantically precise matches (not similar approximations)? → If No: Return "None"
4. Do examples demonstrate the required query pattern or can it be constructed from available schema? → If No: Return "None"
5. All checks pass? → Generate SQL using examples as structural templates

## Common Precision Requirements

**Column Selection Precision**:
- When question specifies full measurement name, use full column label, not abbreviations
- When multiple similar columns exist, verify which captures the exact concept requested
- Lab test measurements and vital sign measurements may share names but live in different tables

**Concept Distinction Examples** (illustrative, not exhaustive):
- Asking for "quality" when only "type" exists → Different attributes, return "None"
- Asking for "performing physician" when only patient or procedure data exists → Different entity, return "None"  
- Asking for "height measurement" when columns include "height" abbreviation and "height (cm)" full label → Use the complete, specific label

**Aggregation Context Examples**:
- "What was the patient's total output in March?" → Multiple output records expected, use SUM
- "What was the patient's first output in March?" → Single record expected, use ORDER BY with LIMIT
- "What was the patient's output?" with no temporal or aggregation context → Check examples for similar patterns

**Time Expression Examples**:
- "since 25 months ago" → `datetime(reference, '-25 month')` as exact rolling window
- "in March this year" → `datetime(time,'start of year') = datetime(reference,'start of year') AND strftime('%m',time) = '03'`
- "on the 15th of this month" → `datetime(time,'start of month') = datetime(reference,'start of month') AND strftime('%d',time) = '15'`
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
# SQL Query Generation from Natural Language

Generate a SQL query by adapting retrieved examples to answer the natural language question. Return **"None"** if the question cannot be answered with available information.

## Core Task Requirements

This is a high-stakes evaluation requiring exact, minimal responses. Your SQL must directly and completely answer the question using only information available in the schema and examples. The question is authoritative; examples serve only as structural templates.

## Two-Phase Validation Protocol

Execute these phases sequentially. Failure at any phase requires immediate return of "None".

### Phase 1: Question Completeness Verification

Verify the question provides all concrete information needed:

- Extract every entity, value, and condition the question references
- Confirm each is explicitly specified without ambiguity or placeholders
- Relative references like "this month" or "last year" are acceptable
- Vague references like "that date", "those patients", or "the procedure" without context are not acceptable

**Return "None" immediately if**: Any specific information needed to answer is missing or too vague to determine.

### Phase 2: Database Feasibility Verification

For each entity or concept in the question, perform exact column matching:

1. **Identify the exact database column** that stores this information
2. **Verify semantic precision**: The column must store the exact concept requested, not a similar or related concept
3. **Document the match**: Column name and table must be cited from schema or examples

**Critical Matching Rules**:

- **Exact attribute matching required**: If the question asks for "quality", a column storing "type" is not acceptable even if both relate to the same entity. Example: "specimen quality" cannot map to a column storing "specimen type" - these are different attributes requiring different columns.

- **Complete entity name matching required**: If the question specifies "or cell saver intake", a column labeled "cell saver" is insufficient. The full entity name must match. Do not use partial string matches or abbreviations when full names are available.

- **Prefer specific over generic labels**: When multiple columns exist, use the most complete and specific one. Example: prefer "height (cm)" over an abbreviation "height" when both exist, as the former is more precise.

- **Table-specific columns**: Some concepts exist in multiple tables with different meanings. Lab tests like "oxygen" or "hemoglobin" use labevents table; vital signs like "temperature" or "blood pressure" use chartevents table; inputs use inputevents table; outputs use outputevents table. Verify you're using the correct table for the concept.

- **Distinguish similar concepts strictly**: The question may introduce subtly different concepts to test precision. You must be extremely strict while judging the feasibility of a concept (i.e., can be mapped to a specific column in the schema). Examples of distinct concepts that are similar, but cannot be mapped to each other and is essentially different:
  - "Specimen quality" (impossible) vs. "specimen name/type" (possible) (different attributes of same entity)
  - "hospital visits" (impossible) vs. "hospital admission" (possible) (different events)
  - "Performing physician" (impossible) vs. "patient" (possible) (different entities)
  - Temperature vs. temperature celsius (different measurement units/standards)

**Return "None" immediately if**:
- Any entity or concept has no exact matching column
- Semantic mismatch exists (similar but not identical concepts)
- Required relationship or join path doesn't exist in schema
- Column stores related but different information than requested

**Only proceed to SQL generation if both phases pass completely.**

## SQL Generation Process

### Step 1: Classify Query Intent

Determine what type of answer the question expects:

- **Existence queries** ("Has there been...", "Is there...", "Does...exist"): Return `SELECT COUNT(*) > 0`
- **Temporal queries** ("When did...", "What time..."): Return timestamp values
- **Count queries** ("How many..."): Return `COUNT()` aggregation
- **Fact queries** ("What is...", "What was..."): Return specific values with careful consideration of aggregation:
  - Use aggregation (SUM, MAX, MIN, AVG) only when explicitly stated or clearly implied by context
  - Singular form with temporal range often implies aggregation: "What was the total output during March" requires SUM
  - Without clear aggregation signal, return individual values or use ordering with LIMIT
  - "What was the dosage prescribed" with multiple prescriptions requires clarification from context: if asking for total, use SUM; if asking for a specific instance, use temporal ordering with LIMIT

### Step 2: Adapt Example Structure

Use retrieved examples as structural templates:

- Examine example questions to find similar query patterns
- Replicate the SQL structure, formatting style, and join patterns from examples
- Replace placeholders with concrete values from entity mappings
- The original question is authoritative - if examples don't perfectly fit, write SQL that directly answers the question rather than forcing an example template

**Structure Adaptation Rules**:
- Include only clauses that directly serve the question's requirements
- Every condition in the question must map to exactly one SQL constraint
- Do not copy irrelevant filters or clauses from examples
- Ensure all aspects of the question are covered in the SQL

### Step 3: Time Expression Handling

Time references require precise interpretation based on patterns demonstrated in the retrieved examples.

**Core principle:** If examples demonstrate datetime functions (`datetime()`, `strftime()`), then time-based filtering operations are supported. Study examples to learn the specific SQL patterns this database uses.

**Relative Time References** (computed dynamically from reference point):

Questions using relative time language require dynamic computation:
- **Period boundaries** (this year, this month, last year): Filter by normalizing timestamps to period starts. Examples will show functions like `datetime(time, 'start of year')` or similar period boundary operations.

- **Combined period + day** (this month/15, last year/March): Apply period boundary filter AND day/month component filter. Examples will show both boundary functions and component extraction functions like `strftime('%d', time)`.

- **Rolling windows** (X days ago, since X months ago): Calculate exact time offsets from reference point - not calendar boundaries. Examples will show offset syntax like `datetime(reference, '-X unit')` where unit is year/month/day.

- **General pattern**: Relative expressions use datetime arithmetic with reference points. Never hardcode relative references to absolute values.

**Absolute Time References** (direct value comparison):

Questions using specific dates/years compare against fixed values:
- **Year/month/date filters**: Extract time components and compare to specific values. Examples will show functions like `strftime('%Y', time) = '2100'` or similar component extraction patterns.

- **General pattern**: Absolute expressions extract components and compare to literals.

**Visit/Encounter Status Indicators**:

Questions about "current" vs "completed" visits use status fields:
- **Ongoing status**: Identified by NULL end/discharge timestamps
- **Completed status**: Identified by NOT NULL end/discharge timestamps
- Examples will show NULL checks on discharge/end time fields

**Duration Calculations**:

When computing time elapsed between two timestamps:
- Convert timestamps to common unit (seconds, days, etc.) using timestamp arithmetic
- Perform subtraction in that unit
- Convert to requested output unit if needed
- Examples will show specific conversion formulas - follow those patterns exactly
- Preserve decimal precision unless examples demonstrate rounding

### Step 4: Aggregation Decision Logic

Apply aggregation functions carefully based on question semantics:

**Apply aggregation when**:
- Explicitly stated: "total", "sum", "average", "maximum", "minimum", "count"
- Singular form implying combination: "What was the dosage prescribed" when multiple prescriptions exist typically means total dosage
- Temporal range with singular measurement: "the output during March" implies total output across the month

**Do not apply aggregation when**:
- Question asks for specific instance: "first", "last", "most recent" → use ORDER BY with LIMIT
- Question asks for individual values: "What are the measurements" (plural)
- Question provides sufficient constraints to return single record without aggregation

**When uncertain**: Examine similar patterns in provided examples for guidance.

## Example-Based Learning

Study the provided examples to understand:
- SQL dialect and formatting conventions (capitalization, spacing, no linebreaks)
- Join patterns for relating tables
- Subquery structures for complex filtering
- Time filtering techniques
- No arbitrary transformations like ROUND() unless demonstrated in examples

## Output Requirements

**Format**:
- If unanswerable: Return exactly the string "None" with no additional text
- If answerable: Return exactly one SQL query with no markdown formatting, no comments, no explanations
- Follow the exact formatting style of examples: no linebreaks within queries, consistent capitalization, spacing patterns

**Style Consistency**:
- Match example SQL formatting precisely
- Do not add arbitrary functions (ROUND, CAST) unless shown in examples
- Do not modify datetime formats unless examples demonstrate the pattern
- Use same quotation style, comma placement, and indentation as examples

## Decision Flowchart

Before generating any SQL, complete this checklist:

1. Does the question provide all concrete information needed? → If No: Return "None"
2. Can each entity/concept map to an exact database column? → If No: Return "None"
3. Are all concepts semantically precise matches (not similar approximations)? → If No: Return "None"
4. Do examples demonstrate the required query pattern or can it be constructed from available schema? → If No: Return "None"
5. All checks pass? → Generate SQL using examples as structural templates

## Common Precision Requirements

**Column Selection Precision**:
- When question specifies full measurement name, use full column label, not abbreviations
- When multiple similar columns exist, verify which captures the exact concept requested
- Lab test measurements and vital sign measurements may share names but live in different tables

**Concept Distinction Examples** (illustrative, not exhaustive):
- Asking for "quality" when only "type" exists → Different attributes, return "None"
- Asking for "performing physician" when only patient or procedure data exists → Different entity, return "None"  
- Asking for "height measurement" when columns include "height" abbreviation and "height (cm)" full label → Use the complete, specific label

**Aggregation Context Examples**:
- "What was the patient's total output in March?" → Multiple output records expected, use SUM
- "What was the patient's first output in March?" → Single record expected, use ORDER BY with LIMIT
- "What was the patient's output?" with no temporal or aggregation context → Check examples for similar patterns

**Time Expression Examples**:
- "since 25 months ago" → `datetime(reference, '-25 month')` as exact rolling window
- "in March this year" → `datetime(time,'start of year') = datetime(reference,'start of year') AND strftime('%m',time) = '03'`
- "on the 15th of this month" → `datetime(time,'start of month') = datetime(reference,'start of month') AND strftime('%d',time) = '15'`
"""

prompt_extension = """
Generate 3 alternative formulations of the given question to improve retrieval coverage. Each formulation should ask for the SAME information but differ meaningfully in structure, abstraction, and framing.

## Core Principle
The 3 formulations should feel like they were written by 3 different people with different communication styles. A reader should notice real differences — not just swapped synonyms.

## Required Variation Strategy
Each of the 3 formulations must use a DIFFERENT strategy from this list:

1. **Implicit/Compressed** — Strip the question to its bare minimum. Drop patient IDs, dates, and qualifiers. Focus only on the core data concept being asked about.
   - "What was the patient's blood pressure at first ICU admission?" → "First ICU blood pressure"
   - "What medication was first given via IV after 05/2100?" → "Initial IV medication"

2. **Narrative/Clinical note style** — Rephrase as something a clinician would write in a note or query a colleague about. More verbose, contextual, clinical tone.
   - "I need to look up the first blood pressure reading recorded when this patient was initially admitted to the ICU."
   - "Trying to find out which IV drug was first prescribed to this patient following a specific date in their stay."

3. **Decomposed/Explicit** — Break the question into its constituent parts or conditions. Make every implicit assumption explicit.
   - "For patient 12345: (1) find all ICU visits, (2) identify the first one, (3) retrieve the blood pressure at that time."
   - "Filter to IV route only. Among those medications, find the one with the earliest prescription date after 05/2100."

## Patient IDs and Dates
- **Strategy 1 (Implicit)**: Always drop patient IDs and specific dates — generalize or omit entirely
- **Strategy 2 (Narrative)**: Keep clinical context but may generalize dates to phrases like "a certain point in their stay"
- **Strategy 3 (Decomposed)**: Keep all specifics — patient ID, dates, filters

## Examples

**Original**: What was patient 12345's blood pressure the first time they visited the ICU?

1. [Implicit] First ICU admission blood pressure reading
2. [Narrative] I'm looking for the blood pressure that was documented when this patient first came into the ICU — the very first admission record.
3. [Decomposed] For patient 12345: identify the earliest ICU admission, then retrieve the blood pressure measurement recorded at that time.

---

**Original**: Tell me the medication patient 10020740 was first prescribed via IV route since 05/2100.

1. [Implicit] Earliest IV medication after a given date
2. [Narrative] I want to know which IV drug this patient was started on first, looking only at prescriptions that came in after a specific point in time in 05/2100.
3. [Decomposed] For patient 10020740: filter all prescriptions to IV route only, restrict to dates on or after 05/2100, return the one with the earliest start date.

---

**Original**: Has patient 10004733 been given any gastric meds since 12/19/2100?

1. [Implicit] Gastric medication administration after a date
2. [Narrative] I need to check whether this patient ever received anything for their stomach — any kind of gastric medication — at any point after mid-December 2100.
3. [Decomposed] For patient 10004733: search medication records for any entry categorized as gastric meds, with administration date strictly after 12/19/2100. Return yes/no and the relevant record if found.

## Output
Return exactly 3 alternative formulations, each clearly using a different strategy. Do not label them — just return the 3 formulations as a plain list.
"""