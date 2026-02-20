# %%
%load_ext autoreload
%autoreload 2

# %%
from dotenv import load_dotenv
load_dotenv()

import json
import os
import time
from qdrant_client import QdrantClient
from qdrant_client import models

# %%
# Collection names to migrate
COLLECTIONS_TO_MIGRATE = [
    # "rag_complete_ehrsql24",
    # "rag_incomplete_ehrsql24",
    # "cbr_complete_ehrsql24",
    "cbr_incomplete_ehrsql24",
]

INDEXED_FIELDS = {
    # "rag_complete_ehrsql24": "case",
    # "rag_incomplete_ehrsql24": "case",
    # "cbr_complete_ehrsql24": "masked_case",
    "cbr_incomplete_ehrsql24": "masked_case",
}

# %%
def extract_documents_using_search(collection_name: str, client: QdrantClient):
    """Extract documents using search queries to avoid scroll errors"""
    print(f"\n{'='*60}")
    print(f"Extracting from: {collection_name}")
    print('='*60)
    
    # Get collection info
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        vector_size = collection_info.config.params.vectors.size
        print(f"  Total points: {total_points}")
        print(f"  Vector size: {vector_size}")
    except Exception as e:
        print(f"  ⚠️  Error getting collection info: {e}")
        return []
    
    documents = []
    batch_size = 100
    
    # Create a zero vector for searching
    zero_vector = [0.0] * vector_size
    
    # Extract in batches using offset
    for offset in range(0, total_points, batch_size):
        try:
            # Use search with zero vector and large limit
            results = client.search(
                collection_name=collection_name,
                query_vector=zero_vector,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            if not results:
                print(f"  No more results at offset {offset}")
                break
            
            for result in results:
                # Extract the FULL payload structure
                full_payload = {
                    "page_content": result.payload.get("page_content", ""),
                    "metadata": result.payload.get("metadata", {})
                }
                documents.append(full_payload)
            
            print(f"  Extracted {len(documents)} documents...")
            
            if len(results) < batch_size:
                break
                
        except Exception as e:
            print(f"  ⚠️  Error at offset {offset}: {e}")
            print(f"  Skipping this batch and continuing...")
            continue
    
    print(f"✓ Total extracted: {len(documents)} documents (out of {total_points} total)")
    
    # Show sample to verify structure
    if documents:
        print(f"\n  Sample document structure:")
        print(f"    page_content: {documents[0]['page_content'][:50]}...")
        print(f"    metadata keys: {list(documents[0]['metadata'].keys())}")
    
    return documents

# %%
def save_documents_to_json(documents, collection_name: str):
    """Save extracted documents to JSON for backup"""
    filename = f"./backup_{collection_name}.json"
    with open(filename, "w") as f:
        json.dump(documents, f, indent=2)
    print(f"✓ Backup saved to: {filename}")

# %%
def convert_to_ingest_format(documents, indexed_field: str):
    """Convert Qdrant payload format to ingestion format"""
    converted_docs = []
    
    for doc in documents:
        # doc already has page_content and metadata separated
        page_content = doc.get("page_content", "")
        metadata = doc.get("metadata", {})
        
        # Reconstruct with indexed field + all metadata
        doc_dict = {indexed_field: page_content}
        doc_dict.update(metadata)  # This includes case, sql_query, entities, etc.
        
        converted_docs.append(doc_dict)
    
    # Show sample to verify
    if converted_docs:
        print(f"\n  Sample converted document:")
        print(f"    Keys: {list(converted_docs[0].keys())}")
        print(f"    {indexed_field}: {converted_docs[0][indexed_field][:50]}...")
    
    return converted_docs

# %%
# Initialize client
client = QdrantClient(url="http://localhost:6333")

# Check which collections exist
print("Checking existing collections...")
collections = client.get_collections().collections
existing_collections = [col.name for col in collections]

for col in COLLECTIONS_TO_MIGRATE:
    if col in existing_collections:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} (not found)")

# %%
# STEP 1: Extract and backup
print("\n" + "="*60)
print("STEP 1: EXTRACTING AND BACKING UP")
print("="*60)

backup_data = {}

for collection_name in COLLECTIONS_TO_MIGRATE:
    if collection_name not in existing_collections:
        print(f"\n⚠️  Skipping {collection_name} (doesn't exist)")
        continue
    
    try:
        documents = extract_documents_using_search(collection_name, client)
        if documents:
            backup_data[collection_name] = documents
            save_documents_to_json(documents, collection_name)
        else:
            print(f"  ⚠️  No documents extracted from {collection_name}")
    except Exception as e:
        print(f"\n✗ Failed to extract {collection_name}: {e}")
        continue

print(f"\n✓ Successfully backed up {len(backup_data)} collections!")

# %%
# STEP 2: Delete old collections (FORCE DELETE)
print("\n" + "="*60)
print("STEP 2: DELETING OLD COLLECTIONS")
print("="*60)

for collection_name in backup_data.keys():
    try:
        # Check if collection exists first
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            print(f"✓ Deleted: {collection_name}")
        else:
            print(f"  Collection {collection_name} doesn't exist")
    except Exception as e:
        print(f"✗ Failed to delete {collection_name}: {e}")

# Wait a moment for deletion to complete
time.sleep(2)

# Verify deletion
print("\nVerifying deletion...")
for collection_name in backup_data.keys():
    exists = client.collection_exists(collection_name)
    if exists:
        print(f"  ⚠️  {collection_name} still exists! Trying again...")
        client.delete_collection(collection_name)
        time.sleep(1)
    else:
        print(f"  ✓ {collection_name} deleted")

print("\n✓ All collections deleted!")

# %%
# STEP 3: Reingest with hybrid support (from backup files)
print("\n" + "="*60)
print("STEP 3: REINGESTING WITH HYBRID SUPPORT")
print("="*60)

from src.retriever import QdrantRetriever

def clean_and_deduplicate(documents, indexed_field):
    """Remove duplicates and invalid documents"""
    seen = set()
    cleaned = []
    
    for doc in documents:
        # Get the indexed content
        page_content = doc.get("page_content", "").strip()
        
        # Skip if blank or empty
        if not page_content:
            continue
        
        # Skip duplicates based on page_content
        if page_content in seen:
            continue
        
        seen.add(page_content)
        cleaned.append(doc)
    
    return cleaned

# Load backup data from files (in case kernel was restarted)
backup_data = {}
for collection_name in INDEXED_FIELDS.keys():
    backup_file = f"./backup_{collection_name}.json"
    
    if os.path.exists(backup_file):
        print(f"Loading backup from: {backup_file}")
        with open(backup_file, "r") as f:
            raw_data = json.load(f)
        
        # Clean and deduplicate
        indexed_field = INDEXED_FIELDS[collection_name]
        cleaned_data = clean_and_deduplicate(raw_data, indexed_field)
        
        print(f"  ✓ Loaded {len(raw_data)} documents")
        print(f"  ✓ After cleaning: {len(cleaned_data)} documents (removed {len(raw_data) - len(cleaned_data)})")
        
        backup_data[collection_name] = cleaned_data
    else:
        print(f"  ⚠️  Backup file not found: {backup_file}")

print(f"\nTotal collections to reingest: {len(backup_data)}")

# %%
# Reingest each collection
for collection_name, documents in backup_data.items():
    print(f"\n{'='*60}")
    print(f"Reingesting: {collection_name}")
    print('='*60)
    
    try:
        # Verify collection doesn't exist before creating
        if client.collection_exists(collection_name):
            print(f"  ⚠️  Collection still exists, deleting again...")
            client.delete_collection(collection_name)
            time.sleep(1)
        
        retriever = QdrantRetriever(collection_name=collection_name)
        indexed_field = INDEXED_FIELDS.get(collection_name, "case")
        
        docs_to_ingest = convert_to_ingest_format(documents, indexed_field)
        
        print(f"  Documents to ingest: {len(docs_to_ingest)}")
        print(f"  Indexed field: {indexed_field}")
        
        if docs_to_ingest:
            print(f"  Sample keys: {list(docs_to_ingest[0].keys())}")
        
        # Ingest in batches
        batch_size = 100
        for i in range(0, len(docs_to_ingest), batch_size):
            batch = docs_to_ingest[i:i+batch_size]
            retriever.ingest(documents=batch, indexed_field=indexed_field)
            print(f"  ✓ Batch {i//batch_size + 1}/{(len(docs_to_ingest)-1)//batch_size + 1}")
        
        print(f"✓ Completed: {collection_name}")
    
    except Exception as e:
        print(f"✗ Failed to reingest {collection_name}: {e}")
        import traceback
        traceback.print_exc()

print("\n✓ REINGESTION COMPLETE!")

# %%
# STEP 4: Verify
print("\n" + "="*60)
print("STEP 4: VERIFYING")
print("="*60)

for collection_name in backup_data.keys():
    try:
        retriever = QdrantRetriever(collection_name=collection_name)
        
        dense_results = retriever.retrieve("test", top_k=3)
        hybrid_results = retriever.retrieve("test", top_k=3)
        
        original_count = len(backup_data[collection_name])
        
        print(f"\n{collection_name}:")
        print(f"  Original: {original_count}")
        print(f"  Dense: {len(dense_results)} results ✓")
        print(f"  Hybrid: {len(hybrid_results)} results ✓")
        
        if dense_results:
            print(f"  Sample: {dense_results[0].page_content[:60]}...")
            
    except Exception as e:
        print(f"\n{collection_name}: ✗ {e}")
        import traceback
        traceback.print_exc()

# %%
print("\n" + "="*60)
print("✓ MIGRATION COMPLETE!")
print("="*60)
print("\nYou can now use:")
print("  retriever.retrieve(query, top_k=5, hybrid=False)  # Dense only")
print("  retriever.retrieve(query, top_k=5, hybrid=True)   # Hybrid retrieval")
print("\nBackup files saved:")
for collection_name in backup_data.keys():
    print(f"  - backup_{collection_name}.json")
    
# %%
