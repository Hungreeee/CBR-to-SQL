# %%
%load_ext autoreload
%autoreload 2

# %%
"""
Evaluation Script for RAG-to-SQL and CBR-to-SQL
Compares performance on MIMICSQL test set across CDB and IDB environments
"""

from dotenv import load_dotenv
load_dotenv()

import json
import pickle
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.utils import query
from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAGtoSQL, CBRtoSQL
from src.retriever import QdrantRetriever
from src.metrics.mimicsql_metrics import logic_form_accuracy, execution_accuracy

from langchain_community.utilities.sql_database import SQLDatabase

# %%
# ========== CONFIGURATION ==========
DATABASE_LOCATION = "./data/TREQS/evaluation/mimic_db/mimic_all.db"
DATABASE_URI = f"sqlite:///{DATABASE_LOCATION}"
RESULTS_DIR = Path("./results/mimicsql/run-11-cbr-cdb-abl-gpt41")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Collection names
COLLECTIONS = {
    "RAG-CDB": "rag_complete",
    "RAG-IDB": "rag_incomplete",
    "CBR-CDB": "cbr_complete",
    "CBR-IDB": "cbr_incomplete",
}
LOOKUP_COLLECTION = "lookup_table"

# Set to False for OpenAI
USE_AZURE = True  

# Select which pipelines to evaluate
EVALUATE = ["CBR-CDB"]
# EVALUATE = ["RAG-CDB", "RAG-IDB", "CBR-CDB", "CBR-IDB"]  # All

print(f"Results will be saved to: {RESULTS_DIR}")
print(f"Evaluating: {EVALUATE}")

# %%
# ========== LOAD DATA ==========
def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

testset = load_jsonl("./data/TREQS/mimicsql_data/mimicsql_natural_v2/test.json")
print(f"✓ Loaded {len(testset)} test examples")

# %%
# ========== INITIALIZE PIPELINES ==========
sql_db = SQLDatabase.from_uri(DATABASE_URI)
generator = AzureAIAgent() if USE_AZURE else OpenAIAgent()
lookup_table = QdrantRetriever(collection_name=LOOKUP_COLLECTION)
fallback_generator = OpenAIAgent()
sql_eval_model = query(DATABASE_LOCATION)

pipelines = {}
for name in EVALUATE:
    if name.startswith("RAG"):
        pipelines[name] = RAGtoSQL(
            retriever=QdrantRetriever(collection_name=COLLECTIONS[name]),
            generator=generator,
            sql_db=sql_db,
        )
    else:  # CBR
        pipelines[name] = CBRtoSQL(
            retriever=QdrantRetriever(collection_name=COLLECTIONS[name]),
            generator=generator,
            sql_db=sql_db,
            lookup_table=lookup_table,
            fallback_generator=fallback_generator,
        )

print(f"✓ Initialized {len(pipelines)} pipelines")

# %%
# ========== EVALUATION FUNCTIONS ==========
def ckpt_step(p):
    return int(p.stem.split("_ckp_")[-1])

def evaluate_pipeline(
    pipeline: RAGtoSQL,
    dataset: List[Dict],
    name: str,
    checkpoint_iters: int = 100,
) -> List[Dict]:
    """Run pipeline on dataset and collect results"""
    results = []
    
    # Try to load the most recent checkpoint
    checkpoint_files = list(RESULTS_DIR.glob(f"{name.lower().replace('-', '_')}_ckp_*.pkl"))
    if checkpoint_files:
        latest_ckpt = max(checkpoint_files, key=ckpt_step)
        with open(latest_ckpt, "rb") as f:
            results = pickle.load(f)
        print(f"📂 Resumed from {latest_ckpt.name}: {len(results)} existing results")
    
    start_idx = len(results)
    error_flag = False
    
    for data in tqdm(
        dataset[start_idx:],
        desc=f"Evaluating {name}",
        initial=start_idx,
        total=len(dataset),
    ):
        question = data["question_refine"]
        gold_sql = data["sql"]
        
        try:
            response = pipeline.handle_request(question)
            results.append({
                "question": question,
                "predicted_sql": response["sql_query"],
                "gold_sql": gold_sql,
                "execution_results": response["execution_results"],
                "retrieved_cases": response["retrieved_cases"],
                "entities": response.get("entities", []),
                "token_usage": response["token_usage"],
            })
        except Exception as e:
            if "aalto" in str(e).lower() or "429" in str(e).lower():
                error_flag = True
                print("Breaking due to rate limit exceeded.")
                break
            results.append({
                "question": question,
                "predicted_sql": "",
                "gold_sql": gold_sql,
                "execution_results": "ERROR",
                "error": str(e),
            })

        # Save checkpoint
        if len(results) % checkpoint_iters == 0 or error_flag:
            ckpt_num = len(results)
            with open(RESULTS_DIR / f"{name.lower().replace('-', '_')}_ckp_{ckpt_num}.pkl", "wb") as f:
                pickle.dump(results, f)
            print(f"  💾 Checkpoint saved: {ckpt_num} results")

    return results

# %%
# ========== LOAD OR RUN EVALUATIONS ==========
all_results = {}

print("\n" + "="*60)
print("LOADING/RUNNING EVALUATIONS")
print("="*60)

for name in EVALUATE:
    filename = f"{name.lower().replace('-', '_')}_results.pkl"
    filepath = RESULTS_DIR / filename
    
    # If final results exist, load them
    if filepath.exists():
        with open(filepath, "rb") as f:
            results = pickle.load(f)
        all_results[name] = results
        print(f"✓ Loaded {name} from disk ({len(results)} results)")
    else:
        # Run evaluation (will auto-resume from latest checkpoint if exists)
        print(f"\n--- Running {name} ---")
        results = evaluate_pipeline(pipelines[name], testset, name)
        all_results[name] = results
        
        # Save final results
        with open(filepath, "wb") as f:
            pickle.dump(results, f)
        print(f"✓ Completed and saved to {filename}")

print(f"\n✓ Total result sets ready: {len(all_results)}")

# %%
# ========== COMPUTE METRICS ==========
print("\n" + "="*60)
print("COMPUTING METRICS")
print("="*60)

all_metrics = {}
for name, results in all_results.items():
    print(f"\n--- {name} ---")
    lf_acc = logic_form_accuracy(results, sql_eval_model)
    print(f"Logical Form Accuracy: {lf_acc}\n\n==========\n\n")
    ex_acc = execution_accuracy(results, sql_eval_model)
    print(f"Execution Accuracy: {ex_acc}")
    
    all_metrics[name] = {
        "lf_accuracy": lf_acc, 
        "ex_accuracy": ex_acc,
    }

# Save metrics
with open(RESULTS_DIR / "metrics.json", "w") as f:
    serializable = {k: {m: v for m, v in metrics.items() if m != "error_idx" and m != "success_idx"} 
                    for k, metrics in all_metrics.items()}
    json.dump(serializable, f, indent=2)
print(f"\n✓ Metrics saved to {RESULTS_DIR / 'metrics.json'}")

# %%
