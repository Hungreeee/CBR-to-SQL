# %%
"""
Safe extraction and reindex of Qdrant CBR collection
"""

from qdrant_client import QdrantClient
from tqdm import tqdm
import json

# Configuration
CBR_COLLECTION = "cbr_incomplete_ehrsql24"
NEW_COLLECTION = CBR_COLLECTION
BATCH_SIZE = 50
QDRANT_URL = "http://localhost:6333"

# Initialize client
client = QdrantClient(url=QDRANT_URL)

# %% 
from tqdm import tqdm
import json

# Step 1: Get actual IDs first (safe, no vectors)
print("Fetching actual IDs from collection...")
all_ids = []
next_page = None
while True:
    try:
        points, next_page = client.scroll(
            collection_name=CBR_COLLECTION,
            limit=50,
            offset=next_page,
            with_payload=False,
            with_vectors=False
        )
        all_ids.extend([p.id for p in points])
        if next_page is None:
            break
    except Exception as e:
        print(f"Scroll failed at page {next_page}: {e}")
        break

print(f"✓ Found {len(all_ids)} IDs")

# Step 2: Retrieve points safely, skip failures
all_points = []
failed_points = []

print("Retrieving points safely (with vectors)...")
for pid in tqdm(all_ids, desc="Fetching points"):
    try:
        point = client.retrieve(
            collection_name=CBR_COLLECTION,
            ids=[pid],
            with_payload=True,
            with_vectors=True
        )
        if point:  # only append if we got a valid point
            all_points.extend(point)
        else:
            failed_points.append(pid)
    except Exception as e:
        failed_points.append(pid)
        print(f"🔥 Retrieval failed for ID {pid}: {e}")

print(f"\n✓ Successfully retrieved {len(all_points)} points")
print(f"⚠️ Failed to retrieve {len(failed_points)} points")

# Save failed points for inspection
with open("failed_points.json", "w") as f:
    json.dump(failed_points, f, indent=2)
