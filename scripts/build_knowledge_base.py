# %%
"""
Case Retention Script (Interactive)
Retains cases for both RAG-to-SQL and CBR-to-SQL pipelines
Supports both CDB (Complete Database) and IDB (Incomplete Database) environments
"""

from dotenv import load_dotenv
load_dotenv()

import json
from tqdm import tqdm
from typing import List, Dict
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAGtoSQL, CBRtoSQL
from src.retriever import QdrantRetriever
from langchain_community.utilities.sql_database import SQLDatabase

from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

np.random.seed(42)

# ========== CONFIGURATION ==========

# %%
DATABASE_URI = "sqlite:///./data/TREQS/evaluation/mimic_db/mimic_all.db"

# Collection names
RAG_CDB_COLLECTION = "rag_complete"      # RAG with all data
RAG_IDB_COLLECTION = "rag_incomplete"    # RAG with clustered data
CBR_CDB_COLLECTION = "cbr_complete"      # CBR with all data
CBR_IDB_COLLECTION = "cbr_incomplete"    # CBR with clustered data
LOOKUP_TABLE_COLLECTION = "lookup_table"

# Clustering parameters for IDB
MIN_CLUSTER_SIZE = 2
CLUSTER_EPSILON = 0.10

# ========== LOAD DATA ==========

# %%
print("\n=== Loading Datasets ===")

trainset = []
with open("./data/TREQS/mimicsql_data/mimicsql_natural_v2/train.json", "r") as f:
    for line in f.readlines():
        json_object = json.loads(line)
        trainset.append(json_object)

testset = []
with open("./data/TREQS/mimicsql_data/mimicsql_natural_v2/test.json", "r") as f:
    for line in f.readlines():
        json_object = json.loads(line)
        testset.append(json_object)

print(f"✓ Loaded {len(trainset)} training examples")
print(f"✓ Loaded {len(testset)} test examples")

# ========== INITIALIZE COMPONENTS ==========

# %%
print("\n=== Initializing Components ===")

sql_db = SQLDatabase.from_uri(DATABASE_URI)
generator = OpenAIAgent()
# generator = AzureAIAgent()  # Alternative

# Note: Lookup table should already be constructed using the lookup table script
lookup_table = QdrantRetriever(collection_name=LOOKUP_TABLE_COLLECTION)

# ========== HELPER FUNCTIONS ==========

# %%
def remove_condition_values(sql: str) -> str:
    """
    Mask entity values in SQL to create abstract templates for clustering
    """
    sql = re.sub(r'=\s*"[^"]*"', '= "VALUE"', sql)
    sql = re.sub(r'([<>]=?)\s*[\d\.]+', r'\1 VALUE', sql)         
    sql = re.sub(r'([<>]=?)\s*"[^"]*"', r'\1 "VALUE"', sql)    
    sql = re.sub(r'IN\s*\(([^)]*)\)', 'IN (VALUE)', sql, flags=re.IGNORECASE)
    return sql


def create_idb_environment(
    trainset: List[Dict],
    min_cluster_size: int = 2,
    cluster_epsilon: float = 0.10
) -> Dict[int, List[Dict]]:
    """
    Create Incomplete Database (IDB) environment using HDBSCAN clustering.
    Returns one representative case per cluster + all outliers.
    """
    print("\n=== Creating IDB Environment via Clustering ===")
    
    # Step 1: Encode masked SQL queries
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    print("Encoding SQL queries...")
    encoded_sql = []
    for data in tqdm(trainset, desc="Encoding"):
        masked_sql = remove_condition_values(data["sql"])
        embeddings = model.encode(masked_sql)
        encoded_sql.append(embeddings)
    
    # Step 2: Cluster using HDBSCAN
    print(f"\nClustering with HDBSCAN (min_cluster_size={min_cluster_size}, epsilon={cluster_epsilon})...")
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_epsilon,
        n_jobs=-1,
    )
    hdb.fit(encoded_sql)
    labels = hdb.labels_
    
    # Step 3: Group by cluster
    label_dict = {}
    for idx, label in enumerate(labels):
        if label not in label_dict:
            label_dict[label] = [trainset[idx]]
        else:
            label_dict[label].append(trainset[idx])
    
    num_clusters = len([l for l in set(labels) if l != -1])
    num_noise = len(label_dict.get(-1, []))
    
    print(f"\n✓ Clustering complete:")
    print(f"  - Clusters found: {num_clusters}")
    print(f"  - Noise points (outliers): {num_noise}")
    print(f"  - Total unique patterns: {num_clusters + num_noise}")
    
    return label_dict


def retain_cases_rag(
    pipeline: RAGtoSQL,
    dataset: List[Dict],
    desc: str = "Retaining cases"
):
    """
    Retain cases for RAG-to-SQL pipeline
    """
    for data in tqdm(dataset, desc=desc):
        question = data["question_refine"]
        sql_query = data["sql"]
        pipeline.retain_case(question, sql_query)


def retain_cases_cbr(
    pipeline: CBRtoSQL,
    dataset: List[Dict],
    desc: str = "Retaining cases"
):
    """
    Retain cases for CBR-to-SQL pipeline (includes entity tagging)
    """
    for data in tqdm(dataset, desc=desc):
        question = data["question_refine"]
        sql_query = data["sql"]
        pipeline.retain_case(question, sql_query)


# ========== ENVIRONMENT 1: CDB (COMPLETE DATABASE) ==========

# %%
print("\n" + "="*60)
print("ENVIRONMENT 1: COMPLETE DATABASE (CDB)")
print("="*60)

# Initialize RAG-CDB pipeline
print("\n--- RAG-to-SQL CDB ---")
rag_cdb_retriever = QdrantRetriever(collection_name=RAG_CDB_COLLECTION)
rag_cdb_pipeline = RAGtoSQL(
    retriever=rag_cdb_retriever,
    generator=generator,
    sql_db=sql_db,
)

# Retain all training cases
print(f"Retaining {len(trainset)} cases for RAG-CDB...")
retain_cases_rag(rag_cdb_pipeline, trainset, desc="RAG-CDB retention")
print("✓ RAG-CDB case retention complete!")

# %%
# Initialize CBR-CDB pipeline
print("\n--- CBR-to-SQL CDB ---")
cbr_cdb_retriever = QdrantRetriever(collection_name=CBR_CDB_COLLECTION)
cbr_cdb_pipeline = CBRtoSQL(
    retriever=cbr_cdb_retriever,
    generator=generator,
    sql_db=sql_db,
    lookup_table=lookup_table,
)

# Retain all training cases (with entity tagging)
print(f"Retaining {len(trainset)} cases for CBR-CDB...")
retain_cases_cbr(cbr_cdb_pipeline, trainset, desc="CBR-CDB retention")
print("✓ CBR-CDB case retention complete!")

# ========== ENVIRONMENT 2: IDB (INCOMPLETE DATABASE) ==========

# %%
print("\n" + "="*60)
print("ENVIRONMENT 2: INCOMPLETE DATABASE (IDB)")
print("="*60)

# Create IDB environment via clustering
label_dict = create_idb_environment(
    trainset,
    min_cluster_size=MIN_CLUSTER_SIZE,
    cluster_epsilon=CLUSTER_EPSILON
)

# %%
# Show cluster statistics
print("\n--- Top 10 Largest Clusters ---")
for label, items in sorted(label_dict.items(), key=lambda x: -len(x[1]))[:10]:
    if label == -1:
        print(f"\n❌ Noise Cluster (-1): {len(items)} items (all will be retained)")
    else:
        print(f"\n✅ Cluster {label}: {len(items)} items (1 representative will be retained)")
    
    # Show sample questions
    for item in items[:3]:
        print(f"   - {item['question_refine']}")
    if len(items) > 3:
        print(f"   ... ({len(items) - 3} more)")

# %%
# Build IDB dataset: 1 representative per cluster + all outliers
print("\n--- Building IDB Dataset ---")
idb_dataset = []

for label, items in label_dict.items():
    if label == -1:
        # Noise: retain all outliers
        idb_dataset.extend(items)
        print(f"  Added {len(items)} noise/outlier cases")
    else:
        # Cluster: retain only the first case as representative
        idb_dataset.append(items[0])

print(f"\n✓ IDB dataset created: {len(idb_dataset)} cases")
print(f"  Original dataset: {len(trainset)} cases")
print(f"  Reduction: {len(trainset) - len(idb_dataset)} cases ({(1 - len(idb_dataset)/len(trainset))*100:.1f}%)")

# %%
# Initialize RAG-IDB pipeline
print("\n--- RAG-to-SQL IDB ---")
rag_idb_retriever = QdrantRetriever(collection_name=RAG_IDB_COLLECTION)
rag_idb_pipeline = RAGtoSQL(
    retriever=rag_idb_retriever,
    generator=generator,
    sql_db=sql_db,
)

# Retain IDB cases
print(f"Retaining {len(idb_dataset)} cases for RAG-IDB...")
retain_cases_rag(rag_idb_pipeline, idb_dataset, desc="RAG-IDB retention")
print("✓ RAG-IDB case retention complete!")

# %%
# Initialize CBR-IDB pipeline
print("\n--- CBR-to-SQL IDB ---")
cbr_idb_retriever = QdrantRetriever(collection_name=CBR_IDB_COLLECTION)
cbr_idb_pipeline = CBRtoSQL(
    retriever=cbr_idb_retriever,
    generator=generator,
    sql_db=sql_db,
    lookup_table=lookup_table,
)

# Retain IDB cases (with entity tagging)
print(f"Retaining {len(idb_dataset)} cases for CBR-IDB...")
print("⚠️  Warning: This will take longer due to entity tagging")
retain_cases_cbr(cbr_idb_pipeline, idb_dataset, desc="CBR-IDB retention")
print("✓ CBR-IDB case retention complete!")

# ========== VISUALIZATION (OPTIONAL) ==========

# %%
# ========== ENVIRONMENT 2: IDB (INCOMPLETE DATABASE) ==========

# %%
print("\n" + "="*60)
print("ENVIRONMENT 2: INCOMPLETE DATABASE (IDB)")
print("="*60)

# Create IDB environment via clustering
# Store all intermediate variables for later visualization
idb_creation_results = create_idb_environment(
    trainset,
    min_cluster_size=MIN_CLUSTER_SIZE,
    cluster_epsilon=CLUSTER_EPSILON
)

# Unpack results
label_dict = idb_creation_results["label_dict"]
hdb = idb_creation_results["hdb"]
encoded_sql = idb_creation_results["encoded_sql"]

# %%
# Show cluster statistics
print("\n--- Top 10 Largest Clusters ---")
for label, items in sorted(label_dict.items(), key=lambda x: -len(x[1]))[:10]:
    if label == -1:
        print(f"\n❌ Noise Cluster (-1): {len(items)} items (all will be retained)")
    else:
        print(f"\n✅ Cluster {label}: {len(items)} items (1 representative will be retained)")
    
    # Show sample questions
    for item in items[:3]:
        print(f"   - {item['question_refine']}")
    if len(items) > 3:
        print(f"   ... ({len(items) - 3} more)")

# %%
# Build IDB dataset: 1 representative per cluster + all outliers
print("\n--- Building IDB Dataset ---")
idb_dataset = []

for label, items in label_dict.items():
    if label == -1:
        # Noise: retain all outliers
        idb_dataset.extend(items)
        print(f"  Added {len(items)} noise/outlier cases")
    else:
        # Cluster: retain only the first case as representative
        idb_dataset.append(items[0])

print(f"\n✓ IDB dataset created: {len(idb_dataset)} cases")
print(f"  Original dataset: {len(trainset)} cases")
print(f"  Reduction: {len(trainset) - len(idb_dataset)} cases ({(1 - len(idb_dataset)/len(trainset))*100:.1f}%)")

# %%
# Initialize RAG-IDB pipeline
print("\n--- RAG-to-SQL IDB ---")
rag_idb_retriever = QdrantRetriever(collection_name=RAG_IDB_COLLECTION)
rag_idb_pipeline = RAGtoSQL(
    retriever=rag_idb_retriever,
    generator=generator,
    sql_db=sql_db,
)

# Retain IDB cases
print(f"Retaining {len(idb_dataset)} cases for RAG-IDB...")
retain_cases_rag(rag_idb_pipeline, idb_dataset, desc="RAG-IDB retention")
print("✓ RAG-IDB case retention complete!")

# %%
# Initialize CBR-IDB pipeline
print("\n--- CBR-to-SQL IDB ---")
cbr_idb_retriever = QdrantRetriever(collection_name=CBR_IDB_COLLECTION)
cbr_idb_pipeline = CBRtoSQL(
    retriever=cbr_idb_retriever,
    generator=generator,
    sql_db=sql_db,
    lookup_table=lookup_table,
)

# Retain IDB cases (with entity tagging)
print(f"Retaining {len(idb_dataset)} cases for CBR-IDB...")
print("⚠️  Warning: This will take longer due to entity tagging")
retain_cases_cbr(cbr_idb_pipeline, idb_dataset, desc="CBR-IDB retention")
print("✓ CBR-IDB case retention complete!")

# ========== VISUALIZATION (OPTIONAL) ==========

# %%
print("\n" + "="*60)
print("OPTIONAL: CLUSTER VISUALIZATION")
print("="*60)

# Ask user if they want to visualize
visualize = input("Do you want to visualize the clusters? (y/n): ").lower() == 'y'

if visualize:
    # Use the variables from create_idb_environment
    labels = hdb.labels_
    encoded_sql_array = np.array(encoded_sql)
    mask = labels != -1
    encoded_filtered = encoded_sql_array[mask]
    labels_filtered = labels[mask]
    
    # Compute cluster means
    cluster_means = []
    cluster_sizes = []
    cluster_ids = []
    
    for label in sorted(set(labels_filtered)):
        cluster_points = encoded_filtered[labels_filtered == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        cluster_means.append(cluster_mean)
        cluster_sizes.append(len(cluster_points))
        cluster_ids.append(label)
    
    cluster_means = np.array(cluster_means)
    cluster_sizes = np.array(cluster_sizes)
    
    # UMAP projection
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="euclidean", random_state=42)
    cluster_embeddings = reducer.fit_transform(cluster_means)
    
    # Normalize sizes
    sizes = 1000 * (cluster_sizes / cluster_sizes.max())
    
    # Plot
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("husl", len(cluster_ids))
    
    for i, (x, y) in enumerate(cluster_embeddings):
        label = cluster_ids[i]
        plt.scatter(
            x, y,
            s=sizes[i],
            color=palette[i],
            alpha=0.6,
            edgecolors='k',
        )
        plt.annotate(
            f"{label}\n({cluster_sizes[i]})",
            (x, y),
            fontsize=8,
            ha='center'
        )
    
    plt.title("HDBSCAN Masked SQL Cluster Summary (IDB Environment)")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig("idb_clusters.png", dpi=150)
    print("✓ Visualization saved to 'idb_clusters.png'")
    plt.show()

# ========== SUMMARY ==========

# %%
print("\n" + "="*60)
print("CASE RETENTION SUMMARY")
print("="*60)
print("\n✓ All pipelines initialized and cases retained!")
print("\nAvailable pipelines:")
print(f"  1. RAG-CDB:  {len(trainset)} cases in '{RAG_CDB_COLLECTION}'")
print(f"  2. RAG-IDB:  {len(idb_dataset)} cases in '{RAG_IDB_COLLECTION}'")
print(f"  3. CBR-CDB:  {len(trainset)} cases in '{CBR_CDB_COLLECTION}'")
print(f"  4. CBR-IDB:  {len(idb_dataset)} cases in '{CBR_IDB_COLLECTION}'")
print("\nNext steps:")
print("  - Run evaluation script to test these pipelines")
print("  - Compare CDB vs IDB performance")
print("  - Compare RAG vs CBR approaches")
print("="*60)

# %%
# Save IDB dataset for reference (optional)
save_idb = input("\nDo you want to save the IDB dataset to disk? (y/n): ").lower() == 'y'

if save_idb:
    with open("idb_dataset.json", "w") as f:
        json.dump(idb_dataset, f, indent=2)
    print("✓ IDB dataset saved to 'idb_dataset.json'")
    
    with open("cluster_info.json", "w") as f:
        cluster_info = {
            str(label): {
                "size": len(items),
                "questions": [item["question_refine"] for item in items[:5]]
            }
            for label, items in label_dict.items()
        }
        json.dump(cluster_info, f, indent=2)
    print("✓ Cluster info saved to 'cluster_info.json'")

# %%