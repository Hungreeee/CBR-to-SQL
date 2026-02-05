# %%
"""
Lookup Table Construction Script (Interactive)
Constructs a semantic search engine over database entities with intelligent column filtering
"""

import ast
import re
import random
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.utilities.sql_database import SQLDatabase
from src.generator import AzureAIAgent, OpenAIAgent
from src.retriever import QdrantRetriever
from src.schema import SemanticRichnessScore

load_dotenv()

# ========== CONFIGURATION ==========

# %%
DATABASE_URI = "sqlite:///./data/ehrsql-2024/data/mimic_iv/mimic_iv.sqlite"
COLLECTION_NAME = "lookup_table_ehrsql24"
SAMPLE_SIZE = 5  # Number of values to sample per column
SEMANTIC_THRESHOLD = 0.5  # Minimum semantic richness score (0-1)
LEXICAL_THRESHOLD = 0.3

# %%
# Initialize components
generator = OpenAIAgent()
sql_db = SQLDatabase.from_uri(DATABASE_URI)
lookup_table = QdrantRetriever(collection_name=COLLECTION_NAME)
lookup_table.reset() # Reset lookup table

print(f"Connected to database: {DATABASE_URI}")
print(f"Available tables: {sql_db.get_usable_table_names()}")

# ========== STEP 1: COLUMN TYPE FILTERING ==========

# %%
def is_text_column(sample_values: List) -> bool:
    """
    Determine if a column contains text values (not IDs/codes/numbers)
    More lenient - focus on filtering out obvious non-text
    """
    if not sample_values:
        return False
    
    non_text_count = 0
    for value in sample_values[:5]:
        if value is None:
            non_text_count += 1
            continue
        if not isinstance(value, str) or len(value) == 0:
            non_text_count += 1
            continue
            
        # Only exclude if it's CLEARLY not text:
        
        # Pure numbers (including decimals)
        if re.match(r'^[\d\.\-\s]+$', value):
            non_text_count += 1
            continue
        
        # UUIDs
        if re.match(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$', value):
            non_text_count += 1
            continue
        
        # Very long random-looking strings (>32 chars, high entropy)
        if len(value) > 32 and re.match(r'^[a-zA-Z0-9]+$', value):
            non_text_count += 1
            continue
    
    # At least 50% should be potentially meaningful text
    # This is LENIENT - we let semantic richness do the heavy filtering
    return non_text_count < len(sample_values[:5]) * 0.5

# %%
# Test the filtering function
print("\n=== Testing Column Type Filtering ===")
test_cases = [
    ("subject_id", ["12345", "67890"]),
    ("icd9_code", ["AOKDOSK250.00", "40SDSDS1.9"]),
    ("short_title", ["Diabetes mellitus super 1.0 melonin", "Hypertension 2-29 type"]),
    ("long_title", ["Diabetes mellitus", "Hypertension"]),
    ("admission_date", ["2019-01-01", "2019-02-15"]),
    ("drug", ["Metformin", "Aspirin"]),
    ("flag_deceased", ["0", "1"]),
]

for col_name, samples in test_cases:
    result = is_text_column(samples)
    print(f"  {col_name}: {'✓ INCLUDE' if result else '✗ EXCLUDE'}")

# ========== STEP 2: SEMANTIC RICHNESS EVALUATION ==========

# %%
def calculate_lexical_diversity(values: List[str]) -> float:
    """
    Simple metric: ratio of unique words to total words
    Higher diversity = more semantically rich
    """
    if not values:
        return 0.0
    
    all_words = []
    for value in values:
        if isinstance(value, str):
            # Tokenize by whitespace and basic punctuation
            words = re.findall(r'\w+', value.lower())
            all_words.extend(words)
    
    if not all_words:
        return 0.0
    
    unique_words = len(set(all_words))
    total_words = len(all_words)
    
    # Normalize: perfect diversity = 1.0, no diversity = 0.0
    diversity = unique_words / total_words if total_words > 0 else 0.0
    
    return diversity

# %%
def evaluate_semantic_richness_llm(
    column_name: str,
    table_name: str,
    sample_values: List[str],
    generator: AzureAIAgent
) -> Dict:
    """Use LLM to evaluate semantic richness - simplified version"""
    formatted_samples = "\n".join([f"  - {v}" for v in sample_values if v])
    
    llm_structured = generator.client.with_structured_output(SemanticRichnessScore)
    
    messages = [
        ("system", "You are a database quality analyzer evaluating semantic richness of column values."),
        ("human", f"""Evaluate the semantic richness of these database values.

Table: {table_name}
Column: {column_name}
Sample values:
{formatted_samples}

Scoring guidelines (0.0 to 1.0):

RICH (0.7-1.0): Descriptive, human-readable text
- Multi-word medical terms: "Diabetes Mellitus Type 2", "Acute Renal Failure"
- Full descriptive names: "Emergency Department", "Intensive Care Unit"
- Complete words: "Metformin", "Emergency", "Elective"

MODERATE (0.4-0.7): Short but meaningful
- Medical abbreviations with context: "ICU", "CABG", "ER"
- Single meaningful words: "Male", "Female", "White", "Hispanic"
- Language/ethnicity codes in context: "ENGL", "SPAN" (if in language column)
- Status terms: "EMERGENCY", "ELECTIVE", "URGENT"

LOW (0.0-0.3): Not semantically useful
- Pure numbers: "123", "45.67"
- Random IDs: "ABC123XYZ", "12345678"
- Meaningless codes: "XJ9K2" 
- Single characters: "M", "F" (unless gender/boolean)

Context matters: 
- "ENGL" in a language column = MODERATE (0.5)
- "URGENT" in admission_type column = MODERATE (0.6)
- "M" in gender column = LOW-MODERATE (0.3-0.4)
- Same values as random IDs = LOW (0.1)

Score based on whether these values would help a user understand the data.""")
    ]
    
    try:
        result = llm_structured.invoke(messages)
        return {
            "score": result.score,
            "reasoning": result.reasoning,
            "examples_analysis": result.examples_analysis,
            "method": "llm"
        }
    except Exception as e:
        print(f"  LLM evaluation failed: {e}")
        return {
            "score": calculate_lexical_diversity(sample_values),
            "reasoning": "Fallback: lexical diversity calculation",
            "method": "lexical"
        }

# %%
def evaluate_semantic_richness_simple(sample_values: List[str]) -> Dict:
    """
    Simple rule-based semantic richness evaluation (no LLM)
    Strict filtering to catch various ID patterns
    """
    if not sample_values:
        return {"score": 0.0, "reasoning": "No values", "method": "simple"}
    
    scores = []
    
    for value in sample_values:
        if not isinstance(value, str) or not value:
            scores.append(0.0)
            continue
        
        value_stripped = value.strip()
        
        # Rule 1: Just numbers = 0.0
        if re.match(r'^[\d\.\-\s]+$', value_stripped):
            scores.append(0.0)
            continue
        
        # Rule 2: High digit ratio (>40%) = 0.1
        digit_ratio = sum(c.isdigit() for c in value_stripped) / len(value_stripped)
        if digit_ratio > 0.4:
            scores.append(0.1)
            continue
        
        # Rule 3: Mixed alphanumeric without spaces = 0.1
        if ' ' not in value_stripped:
            has_digit = any(c.isdigit() for c in value_stripped)
            has_letter = any(c.isalpha() for c in value_stripped)
            if has_digit and has_letter:
                scores.append(0.1)
                continue
        
        # Rule 4: Multi-word = high score
        word_count = len(value_stripped.split())
        if word_count >= 3:
            scores.append(0.9)
        elif word_count == 2:
            scores.append(0.7)
        elif len(value_stripped) >= 5:
            scores.append(0.6)
        else:
            scores.append(0.3)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "score": avg_score,
        "reasoning": f"Avg of {len(scores)} scores",
        "method": "simple"
    }

# %%
def sample_column_values(
    sql_db: SQLDatabase,
    table_name: str,
    column_name: str,
    sample_size: int = 10
) -> List:
    """
    Randomly sample distinct values from a column
    """
    column_name = column_name.strip('"').strip("'")
    table_name = table_name.strip('"').strip("'")

    query = f"""
    SELECT DISTINCT "{column_name}"
    FROM "{table_name}"
    WHERE "{column_name}" IS NOT NULL
    LIMIT {sample_size}
    """
    
    try:
        result = sql_db.run(query)
        
        # Debug: check if truncation is happening
        print(f"      [DEBUG] Result length: {len(result)} chars")
        print(f"      [DEBUG] Full result: {result}")
        
        # Parse using ast.literal_eval
        parsed = ast.literal_eval(result)
        values = [str(row[0]) for row in parsed if row and row[0] is not None]
        
        print(f"      [DEBUG] Sample values: {values[:10]}")
        
        # Random sample
        if len(values) > sample_size:
            values = random.sample(values, sample_size)
        
        return values
    except Exception as e:
        print(f"  Error sampling {table_name}.{column_name}: {e}")
        return []

# %%
def construct_lookup_table(
    sql_db: SQLDatabase,
    lookup_table: QdrantRetriever,
    generator: AzureAIAgent = None,
    use_llm: bool = False,
    sample_size: int = 10,
    simple_threshold: float = 0.3,  # Lower threshold for simple filtering
    llm_threshold: float = 0.6       # Higher threshold for LLM filtering
) -> Dict:
    """
    Main pipeline to construct the lookup table with two-stage filtering
    
    Stage 1: Simple rule-based filtering (fast, removes obvious junk)
    Stage 2: LLM-based filtering (slow, for nuanced decisions)
    
    Returns statistics about the construction process
    """
    stats = {
        "total_columns": 0,
        "text_columns": 0,
        "simple_filtered_columns": 0,
        "semantically_rich_columns": 0,
        "total_entities": 0,
        "skipped_columns": [],
        "included_columns": []
    }
    
    datapoints = []
    
    # Get all tables
    tables = sql_db.get_usable_table_names()
    
    print(f"\n{'='*60}")
    print(f"CONSTRUCTING LOOKUP TABLE (TWO-STAGE FILTERING)")
    print(f"{'='*60}")
    print(f"Tables to process: {len(tables)}")
    print(f"Stage 1 - Simple threshold: {simple_threshold}")
    if use_llm:
        print(f"Stage 2 - LLM threshold: {llm_threshold}")
    else:
        print(f"Stage 2 - LLM: DISABLED (only using simple filtering)")
    print(f"{'='*60}\n")
    
    for table in tables:
        print(f"\nProcessing table: {table}")
        
        # Get table info
        try:
            table_info = sql_db.get_table_info_no_throw([table])
            # Extract column names (simple parsing)
            columns = []
            for line in table_info.split('\n'):
                if '\t' in line or '  ' in line:
                    parts = re.split(r'\s+', line.strip())
                    if parts and parts[0] and not parts[0].startswith('CREATE'):
                        columns.append(parts[0])
        except Exception as e:
            print(f"  Error getting table info: {e}")
            continue
        
        for column in columns:
            stats["total_columns"] += 1
            
            # Step 1: Sample values
            sample_values = sample_column_values(sql_db, table, column, sample_size)
            
            if not sample_values:
                stats["skipped_columns"].append({
                    "table": table,
                    "column": column,
                    "reason": "no_values"
                })
                continue
            
            # Step 2: Text column filtering (basic type check)
            if not is_text_column(sample_values):
                stats["skipped_columns"].append({
                    "table": table,
                    "column": column,
                    "reason": "not_text_column"
                })
                print(f"  ✗ {column}: Not a descriptive text column (type check)")
                continue
            
            stats["text_columns"] += 1
            
            # Step 3a: SIMPLE semantic richness evaluation (fast filtering)
            simple_richness = evaluate_semantic_richness_simple(sample_values)
            simple_score = simple_richness["score"]
            
            if simple_score < simple_threshold:
                stats["skipped_columns"].append({
                    "table": table,
                    "column": column,
                    "reason": f"simple_filter_failed ({simple_score:.2f})"
                })
                print(f"  ✗ {column}: Failed simple filter (score: {simple_score:.2f})")
                continue
            
            stats["simple_filtered_columns"] += 1
            print(f"  ✓ {column}: Passed simple filter (score: {simple_score:.2f})")
            
            # Step 3b: LLM semantic richness evaluation (nuanced filtering)
            final_score = simple_score
            final_method = "simple"
            
            if use_llm and generator:
                print(f"      → Running LLM evaluation...")
                llm_richness = evaluate_semantic_richness_llm(
                    column, table, sample_values, generator
                )
                llm_score = llm_richness["score"]
                
                # Use LLM score if successful
                if llm_richness["method"] == "llm":
                    final_score = llm_score
                    final_method = "llm"
                    print(f"      → LLM score: {llm_score:.2f}")
                else:
                    print(f"      → LLM failed, using simple score: {simple_score:.2f}")
                
                # Check against LLM threshold
                if final_score < llm_threshold:
                    stats["skipped_columns"].append({
                        "table": table,
                        "column": column,
                        "reason": f"llm_filter_failed ({final_score:.2f})",
                        "simple_score": simple_score,
                        "llm_score": llm_score if final_method == "llm" else None
                    })
                    print(f"  ✗ {column}: Failed LLM filter (score: {final_score:.2f})")
                    continue
            
            # Column passed all filters!
            stats["semantically_rich_columns"] += 1
            stats["included_columns"].append({
                "table": table,
                "column": column,
                "richness_score": final_score,
                "evaluation_method": final_method,
                "simple_score": simple_score,
                "sample_count": len(sample_values)
            })
            
            print(f"  ✓✓ {column}: INCLUDED (final score: {final_score:.2f}, method: {final_method})")
            
            # Step 4: Get all distinct values for this column
            query = f'SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL'

            try:
                result = sql_db.run(query)
                
                # Parse using ast.literal_eval
                parsed = ast.literal_eval(result)
                values = [str(row[0]) for row in parsed if row and row[0] is not None]
                
                # Add to datapoints
                for entity in values:
                    if entity and entity.strip():
                        datapoints.append({
                            "entity": entity.strip(),
                            "table": table,
                            "column": column,
                        })
                        stats["total_entities"] += 1
                
                print(f"      → Added {len(values)} entities")
                
            except Exception as e:
                print(f"      Error extracting entities: {e}")
    
    # Ingest into lookup table
    if datapoints:
        print(f"\n{'='*60}")
        print(f"INGESTING INTO LOOKUP TABLE")
        print(f"{'='*60}")
        print(f"Total entities to ingest: {len(datapoints)}")
        
        lookup_table.ingest(datapoints, "entity")
        print("✓ Ingestion complete!")
    
    return stats

# ========== STEP 4: RUN CONSTRUCTION ==========

# %%
# Run the construction with LLM-based evaluation (slower, more accurate)
stats = construct_lookup_table(
    sql_db=sql_db,
    lookup_table=lookup_table,
    generator=generator,
    use_llm=True,
    sample_size=SAMPLE_SIZE,
    simple_threshold=LEXICAL_THRESHOLD,
    llm_threshold=SEMANTIC_THRESHOLD
)

# ========== STEP 5: DISPLAY RESULTS ==========

# %%
print(f"\n{'='*60}")
print("CONSTRUCTION SUMMARY")
print(f"{'='*60}")
print(f"Total columns analyzed: {stats['total_columns']}")
print(f"└─ Text columns (type check): {stats['text_columns']}")
print(f"   └─ Passed simple filter: {stats['simple_filtered_columns']}")
print(f"      └─ Passed LLM filter: {stats['semantically_rich_columns']}")
print(f"\nTotal entities indexed: {stats['total_entities']}")
print(f"\nIncluded columns ({len(stats['included_columns'])}):")

for col_info in stats['included_columns']:
    method_emoji = "[LLM]" if col_info['evaluation_method'] == 'llm' else "[FALLBACK]"
    print(f"  {method_emoji} {col_info['table']}.{col_info['column']} "
          f"(score: {col_info['richness_score']:.2f}, "
          f"simple: {col_info['simple_score']:.2f})")

# ========== STEP 6: TEST THE LOOKUP TABLE ==========

# %%
print(f"\n{'='*60}")
print("TESTING LOOKUP TABLE")
print(f"{'='*60}")

test_queries = [
    "diabetes",
    "metformin",
    "MRI",
    "sepsis",
    "emergency",
]

for query in test_queries:
    results = lookup_table.retrieve(query, top_k=5)
    print(f"\nQuery: '{query}'")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. '{result.page_content}' "
              f"(Table: {result.metadata.get('table')}, "
              f"Column: {result.metadata.get('column')})")

# %%