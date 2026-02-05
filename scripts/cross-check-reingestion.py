# %%
"""
Cross-check backup data and fill missing cases for CBR-CDB
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
from tqdm import tqdm

from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import CBRtoSQL
from src.retriever import QdrantRetriever
from langchain_community.utilities.sql_database import SQLDatabase

# %%
# Configuration
DATABASE_URI = "sqlite:///./data/ehrsql-2024/data/mimic_iv/mimic_iv.sqlite"
CBR_CDB_COLLECTION = "cbr_complete_ehrsql24"
LOOKUP_TABLE_COLLECTION = "lookup_table_ehrsql24"

# %%
# Load training set
print("Loading training set...")
trainset = []
with open("./data/ehrsql-2024/data/mimic_iv/train/annotated.json", "r") as f:
    trainset = json.loads(f.read())
print(f"✓ Loaded {len(trainset)} training examples")

# %%
# Load backup data
backup_file = "./backup_cbr_complete_ehrsql24.json"
print(f"\nLoading backup from {backup_file}...")
with open(backup_file, "r") as f:
    backup_data = json.load(f)
print(f"✓ Loaded {len(backup_data)} backup documents")

# %%
# Extract and clean questions from backup
backup_questions = set()
valid_backup_docs = []

for doc in backup_data:
    # Get page_content and case
    page_content = doc.get("page_content", "").strip()
    question = doc.get("metadata", {}).get("case", "").strip()
    
    # Skip if both are empty
    if not page_content and not question:
        continue
    
    # Skip if question is empty (invalid document)
    if not question:
        continue
    
    # Add to valid documents and questions
    valid_backup_docs.append(doc)
    backup_questions.add(question)

print(f"\n=== Backup Cleaning Results ===")
print(f"Original backup: {len(backup_data)} documents")
print(f"After cleaning: {len(valid_backup_docs)} documents")
print(f"Removed: {len(backup_data) - len(valid_backup_docs)} invalid/blank documents")
print(f"Unique questions: {len(backup_questions)}")

# Update backup_data to use only valid documents
backup_data = valid_backup_docs

# %%
# Find missing questions
train_questions = {data["question"].strip() for data in trainset}
missing_questions = train_questions - backup_questions

print(f"\n=== Cross-check Results ===")
print(f"Training set: {len(train_questions)} questions")
print(f"Backup set: {len(backup_questions)} questions")
print(f"Missing: {len(missing_questions)} questions")

if len(missing_questions) == 0:
    print("\n✓ No missing cases! Backup is complete.")
else:
    print(f"\n⚠️  {len(missing_questions)} cases need to be processed")

# %%
# Build missing dataset
if len(missing_questions) > 0:
    print("\nBuilding dataset for missing cases...")
    missing_dataset = []
    
    for data in trainset:
        if data["question"].strip() in missing_questions:
            missing_dataset.append(data)
    
    print(f"✓ Created dataset with {len(missing_dataset)} missing cases")
    
    # Show samples
    print("\nSample missing questions:")
    for i, data in enumerate(missing_dataset[:5]):
        print(f"  {i+1}. {data['question'][:80]}...")

# %%
# Run source discovery on missing cases
if len(missing_questions) > 0:
    print("\n=== Running Source Discovery ===")
    
    # Initialize components
    sql_db = SQLDatabase.from_uri(DATABASE_URI)
    generator = OpenAIAgent()
    fallback_generator = AzureAIAgent()
    lookup_table = QdrantRetriever(collection_name=LOOKUP_TABLE_COLLECTION)
    
    # Initialize CBR pipeline (temporary, just for source discovery)
    temp_retriever = QdrantRetriever(collection_name="temp_collection")
    cbr_pipeline = CBRtoSQL(
        retriever=temp_retriever,
        generator=generator,
        sql_db=sql_db,
        lookup_table=lookup_table,
        fallback_generator=fallback_generator
    )
    
    # Process missing cases
    processed_docs = []
    errors = []
    
    for data in tqdm(missing_dataset, desc="Processing missing cases"):
        question = data["question"]
        sql_query = data["query"]
        
        try:
            # Run source discovery
            masked_question, entities = cbr_pipeline.source_discovery(question)
            
            # Create document
            doc = {
                "masked_case": masked_question,
                "case": question,
                "sql_query": sql_query,
                "entities": entities
            }
            processed_docs.append(doc)
            
        except Exception as e:
            print(f"\n⚠️  Error for question: {question[:60]}...")
            print(f"   Error: {str(e)}")
            
            # Fallback: no entities
            doc = {
                "masked_case": question,
                "case": question,
                "sql_query": sql_query,
                "entities": []
            }
            processed_docs.append(doc)
            errors.append({"question": question, "error": str(e)})
    
    print(f"\n✓ Processed {len(processed_docs)} missing cases")
    if errors:
        print(f"⚠️  {len(errors)} cases had errors (using fallback)")

# %%
# Convert backup data to document format
if len(missing_questions) > 0:
    print("\n=== Preparing Complete Dataset ===")
    
    # Convert backup to proper format
    backup_docs = []
    for doc in backup_data:
        backup_docs.append({
            "masked_case": doc.get("page_content", ""),
            "case": doc.get("metadata", {}).get("case", ""),
            "sql_query": doc.get("metadata", {}).get("sql_query", ""),
            "entities": doc.get("metadata", {}).get("entities", [])
        })
    
    # Combine backup + newly processed
    complete_dataset = backup_docs + processed_docs
    
    print(f"Backup documents: {len(backup_docs)}")
    print(f"New documents: {len(processed_docs)}")
    print(f"Total documents: {len(complete_dataset)}")

# %%
# Deduplicate based on case (question)
if len(missing_questions) > 0:
    print("\n=== Deduplicating Dataset ===")
    
    seen_cases = set()
    deduplicated_dataset = []
    
    for doc in complete_dataset:
        case = doc.get("case", "").strip()
        if case and case not in seen_cases:
            seen_cases.add(case)
            deduplicated_dataset.append(doc)
    
    print(f"Before deduplication: {len(complete_dataset)}")
    print(f"After deduplication: {len(deduplicated_dataset)}")
    print(f"Duplicates removed: {len(complete_dataset) - len(deduplicated_dataset)}")
    
    complete_dataset = deduplicated_dataset

# %%
# Save complete dataset
if len(missing_questions) > 0:
    output_file = "./cbr_complete_dataset_fixed.json"
    print(f"\n=== Saving Complete Dataset ===")
    print(f"Saving to: {output_file}")
    
    with open(output_file, "w") as f:
        json.dump(complete_dataset, f, indent=2)
    
    print(f"✓ Saved {len(complete_dataset)} documents")
    
    # Save error log if any
    if errors:
        error_file = "./cbr_processing_errors.json"
        with open(error_file, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"✓ Saved error log to {error_file}")

# %%
# Reingest complete dataset
if len(missing_questions) > 0:
    confirm = input(f"\nReingest {len(complete_dataset)} documents to {CBR_CDB_COLLECTION}? (y/n): ")
    
    if confirm.lower() == 'y':
        print("\n=== Reingesting Complete Dataset ===")
        
        from qdrant_client import QdrantClient
        
        # Delete old collection
        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists(CBR_CDB_COLLECTION):
            print(f"Deleting old collection: {CBR_CDB_COLLECTION}")
            client.delete_collection(CBR_CDB_COLLECTION)
        
        # Initialize new retriever
        retriever = QdrantRetriever(collection_name=CBR_CDB_COLLECTION)
        
        # Ingest in batches
        batch_size = 100
        for i in tqdm(range(0, len(complete_dataset), batch_size), desc="Ingesting"):
            batch = complete_dataset[i:i+batch_size]
            retriever.ingest(documents=batch, indexed_field="masked_case")
        
        print(f"\n✓ Reingestion complete!")
        print(f"Collection: {CBR_CDB_COLLECTION}")
        print(f"Total documents: {len(complete_dataset)}")
    else:
        print("\nSkipped reingestion. Dataset saved to file for manual review.")

# %%
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Training set: {len(trainset)} questions")
print(f"Original backup: {len(backup_data)} documents")
print(f"Missing cases: {len(missing_questions)}")
if len(missing_questions) > 0:
    print(f"Complete dataset: {len(complete_dataset)} documents")
    print(f"Saved to: ./cbr_complete_dataset_fixed.json")
print("="*60)
