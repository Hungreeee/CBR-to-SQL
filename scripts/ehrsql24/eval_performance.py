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

from src.utils import query, stratified_sample
from src.configs import RAGConfig
from src.generator import AzureAIAgent, OpenAIAgent
from src.rag_pipeline import RAGtoSQL, CBRtoSQL
from src.retriever import QdrantRetriever
from src.metrics.ehrsql24_metrics import logic_form_accuracy, execution_accuracy, compute_rs_variants

from langchain_community.utilities.sql_database import SQLDatabase

# ========== CONFIGURATION ==========

# %%
DATABASE_LOCATION = "./data/ehrsql-2024/data/mimic_iv/mimic_iv.sqlite"
DATABASE_URI = f"sqlite:///{DATABASE_LOCATION}"
RESULTS_DIR = Path("./results/ehrsql24/run-13-cbr-idb-brittle")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Collection names
COLLECTIONS = {
    "RAG-CDB": "rag_complete_ehrsql24",
    "RAG-IDB": "rag_incomplete_ehrsql24",
    "CBR-CDB": "cbr_complete_ehrsql24",
    "CBR-IDB": "cbr_incomplete_ehrsql24",
}
LOOKUP_COLLECTION = "lookup_table_ehrsql24"

# Model selection 
USE_AZURE = True  # Set to False for OpenAI

# Select which pipelines to evaluate
EVALUATE = ["CBR-IDB"]
# EVALUATE = ["RAG-CDB", "RAG-IDB", "CBR-CDB", "CBR-IDB"]  # All

print(f"Results will be saved to: {RESULTS_DIR}")
print(f"Evaluating: {EVALUATE}")

# %%
# ========== LOAD DATA ==========
def load_json(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, "r") as f:
        data = json.loads(f.read())
    return data

testset = load_json("./data/ehrsql-2024/data/mimic_iv/test/annotated.json")
print(f"✓ Loaded {len(testset)} test examples")

# %%
# # Stratified sampling for testing
# testset = stratified_sample(testset, p=1/3, label=False)
# len(testset)

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
            config=RAGConfig(dataset="ehrsql24")
        )
    else:  # CBR
        pipelines[name] = CBRtoSQL(
            retriever=QdrantRetriever(collection_name=COLLECTIONS[name]),
            generator=generator,
            sql_db=sql_db,
            lookup_table=lookup_table,
            fallback_generator=fallback_generator,
            config=RAGConfig(dataset="ehrsql24")
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
        question = data["question"]
        gold_sql = data["query"]
        
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
    rs_acc = compute_rs_variants(results, sql_eval_model)
    print(f"Reliability Score: {rs_acc}")
    
    all_metrics[name] = {
        "lf_accuracy": lf_acc, 
        "ex_accuracy": ex_acc,
        "rs_accuracy": rs_acc
    }

# Save metrics
with open(RESULTS_DIR / "metrics.json", "w") as f:
    serializable = {k: {m: v for m, v in metrics.items() if m != "error_idx" and m != "success_idx"} 
                    for k, metrics in all_metrics.items()}
    json.dump(serializable, f, indent=2)
print(f"\n✓ Metrics saved to {RESULTS_DIR / 'metrics.json'}")

# %%
