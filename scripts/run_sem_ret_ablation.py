"""
Unattended runner for the semantic-tagging ablation (refined_tagging=False: generic
tags for masking/retrieval, lookup table still used to locate entities).

Runs one independent chain per dataset:

    ingestion  ->  sanity gate  ->  eval

Ingestion is either
  --wait  DATASET   an ingestion job somebody else is already running (e.g. in a
                    notebook kernel); wait until its collection holds every
                    training case, or
  --build DATASET   run build_knowledge_base_sem_ret.py here.

The eval only starts if the ingestion finished: the collection must hold exactly
one point per training case. Individual cases whose entity extraction failed are
tolerated (retain_case stores them unmasked), unless there are so many that the
extraction was clearly broken.

Models are pinned inside the scripts (MODEL = ...) and verified here before
anything starts:  builds -> gpt-4.1-mini on OpenAI,  evals -> see "eval_on" in
DATASETS below.

Usage (from the repo root, with the project venv):
    python scripts/run_sem_ret_ablation.py check  --wait ehrsql24 --build mimicsql
    python scripts/run_sem_ret_ablation.py ping        # one live call per model/provider
    python scripts/run_sem_ret_ablation.py run    --wait ehrsql24 --build mimicsql
    python scripts/run_sem_ret_ablation.py run    --wait mimicsql ehrsql24 --sequential
    python scripts/run_sem_ret_ablation.py status
    python scripts/run_sem_ret_ablation.py stop

Everything is logged under logs/sem_ret/:
    orchestrator.log        what the runner decided and when
    status.json             current state of each chain
    <dataset>_build.log     output of the build script
    <dataset>_eval.log      output of the eval script
"""

import argparse
import atexit
import ctypes
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
LOG_DIR = ROOT / "logs" / "sem_ret"
PID_FILE = LOG_DIR / "orchestrator.pid"
STATUS_FILE = LOG_DIR / "status.json"
QDRANT_URL = "http://localhost:6333"

# ========== WHAT EACH STAGE MUST RUN ON ==========
BUILD_MODEL, BUILD_GENERATOR = "gpt-4.1-mini", "OpenAIAgent"
# Evals are set per dataset in DATASETS as "eval_on": (MODEL, USE_AZURE)

# ========== TIMING / LIMITS ==========
POLL_SECONDS = 30
SETTLE_SECONDS = 60                 # collection must stay at the target this long
STALL_SECONDS = 45 * 60             # external ingestion: give up after no progress
IDLE_CHECK_SECONDS = 90             # before a build: nobody else may be writing
BUILD_MAX_ATTEMPTS = 3
BUILD_RETRY_SECONDS = 5 * 60
EVAL_RETRY_SECONDS = 15 * 60        # wait after a rate-limit break / crash
QUOTA_MARGIN_SECONDS = 5 * 60       # extra wait after a quota says it is replenished
EVAL_MAX_STUCK_ATTEMPTS = 3         # consecutive attempts without new results
MAX_EMPTY_ENTITY_FRACTION = 0.20    # refined baselines: 0.05% (mimicsql), 2.1% (ehrsql24)
MAX_ERROR_ROW_FRACTION = 0.10

RESULTS_NAME = "cbr_cdb_results.pkl"    # EVALUATE = ["CBR-CDB"]
CHECKPOINT_GLOB = "cbr_cdb_ckp_*.pkl"


def _count_jsonl(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def _count_ehrsql24(split_dir: Path) -> int:
    """Same resolution order as load_split() in the ehrsql24 scripts."""
    annotated = split_dir / "annotated.json"
    if annotated.exists():
        return len(json.loads(annotated.read_text(encoding="utf-8")))
    data = json.loads((split_dir / "data.json").read_text(encoding="utf-8"))
    return len(data["data"] if isinstance(data, dict) and "data" in data else data)


DATASETS = {
    "mimicsql": {
        "build": ROOT / "scripts/mimicsql/build_knowledge_base_sem_ret.py",
        "eval": ROOT / "scripts/mimicsql/eval_performance_sem_ret.py",
        "n_train": lambda: _count_jsonl(ROOT / "data/TREQS/mimicsql_data/mimicsql_natural_v2/train.json"),
        "n_test": lambda: _count_jsonl(ROOT / "data/TREQS/mimicsql_data/mimicsql_natural_v2/test.json"),
        "eval_on": ("gpt-4o", "True"),
    },
    "ehrsql24": {
        "build": ROOT / "scripts/ehrsql24/build_knowledge_base_sem_ret.py",
        "eval": ROOT / "scripts/ehrsql24/eval_performance_sem_ret.py",
        "n_train": lambda: _count_ehrsql24(ROOT / "data/ehrsql-2024/data/mimic_iv/train"),
        "n_test": lambda: _count_ehrsql24(ROOT / "data/ehrsql-2024/data/mimic_iv/test"),
        # Direct OpenAI: the Azure gateway quota ran out part-way through this eval
        "eval_on": ("gpt-4o-2024-08-06", "False"),
    },
}

log = logging.getLogger("sem_ret")


class ChainFailed(Exception):
    """A chain cannot continue; its eval must not run."""


# ========== READ SETTINGS OUT OF THE SCRIPTS ==========
def _grab(path: Path, pattern: str, what: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.M)
    if not match:
        raise RuntimeError(f"{path.name}: could not find {what}")
    return match.group(1)


def resolve(dataset: str) -> Dict:
    """Settings for one dataset, read from its scripts so the two cannot drift."""
    cfg = DATASETS[dataset]
    build, evals = cfg["build"], cfg["eval"]
    return {
        "dataset": dataset,
        "build": build,
        "eval": evals,
        "collection": _grab(build, r'^CBR_CDB_SEM_RET_COLLECTION\s*=\s*"([^"]+)"', "CBR_CDB_SEM_RET_COLLECTION"),
        "eval_collection": _grab(evals, r'^\s*"CBR-CDB":\s*"([^"]+)"', 'the "CBR-CDB" collection'),
        "results_dir": ROOT / _grab(evals, r'^RESULTS_DIR\s*=\s*Path\("([^"]+)"\)', "RESULTS_DIR"),
        "build_model": _grab(build, r'^MODEL\s*=\s*"([^"]+)"', "MODEL"),
        "build_generator": _grab(build, r"^generator\s*=\s*(\w+)\(", "generator"),
        "eval_model": _grab(evals, r'^MODEL\s*=\s*"([^"]+)"', "MODEL"),
        "eval_use_azure": _grab(evals, r"^USE_AZURE\s*=\s*(True|False)", "USE_AZURE"),
        "eval_pipelines": re.sub(r"\s", "", _grab(evals, r"^EVALUATE\s*=\s*(\[[^\]]*\])", "EVALUATE")),
        "eval_on": cfg["eval_on"],
        "n_train": cfg["n_train"](),
        "n_test": cfg["n_test"](),
    }


# ========== QDRANT ==========
def qdrant_count(collection: str, only_empty_entities: bool = False) -> Optional[int]:
    """Exact point count; None if the collection does not exist or Qdrant is unreachable."""
    body: Dict = {"exact": True}
    if only_empty_entities:
        body["filter"] = {"must": [{"is_empty": {"key": "metadata.entities"}}]}
    request = urllib.request.Request(
        f"{QDRANT_URL}/collections/{collection}/points/count",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())["result"]["count"]
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None


# ========== STATUS ==========
_status_lock = threading.Lock()
_status: Dict = {"chains": {}}


def set_status(dataset: str, state: str, detail: str = "", **extra) -> None:
    with _status_lock:
        entry = _status["chains"].setdefault(dataset, {})
        entry.update(state=state, detail=detail, updated=datetime.now().isoformat(timespec="seconds"), **extra)
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(_status, indent=2), encoding="utf-8")
        os.replace(tmp, STATUS_FILE)
    log.info("[%s] %s%s", dataset, state, f" -- {detail}" if detail else "")


# ========== CHILD PROCESSES ==========
def run_child(dataset: str, script: Path, log_path: Path, title: str) -> int:
    """Run one of the cell scripts in the project venv; returns its exit code."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"   # the scripts print non-ASCII; stdout is a file here
    env["PYTHONUNBUFFERED"] = "1"
    env["TQDM_MININTERVAL"] = "30"      # keep progress bars from flooding the log
    with open(log_path, "ab") as out:
        out.write(f"\n{'=' * 70}\n{datetime.now():%Y-%m-%d %H:%M:%S}  {title}\n{'=' * 70}\n".encode("utf-8"))
        out.flush()
        process = subprocess.Popen(
            [str(PYTHON), "-u", str(Path(__file__).resolve()), "exec", str(script)],
            cwd=str(ROOT), env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        with _status_lock:
            _status["chains"].setdefault(dataset, {})["child_pid"] = process.pid
        log.info("[%s] started %s (pid %s) -> %s", dataset, script.name, process.pid, log_path.name)
        code = process.wait()
    log.info("[%s] %s exited with code %s", dataset, script.name, code)
    return code


def exec_script(script: str) -> None:
    """Child entry point: run a `# %%` cell script as a plain program.

    The scripts are written for the interactive window, so the repo root and
    src/ go on sys.path (src modules import each other by bare name) and the
    IPython-only autoreload magics are blanked out, keeping line numbers intact.
    """
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    source = Path(script).read_text(encoding="utf-8")
    source = re.sub(r"(?m)^%(load_ext|autoreload)\b.*$", "", source)
    sys.argv = [script]
    exec(compile(source, script, "exec"), {"__name__": "__main__", "__file__": script})


# ========== INGESTION ==========
def wait_for_external_ingestion(s: Dict) -> None:
    """Block until somebody else's ingestion job has stored every training case."""
    dataset, collection, target = s["dataset"], s["collection"], s["n_train"]
    last_count, last_change = None, time.time()
    while True:
        count = qdrant_count(collection)
        if count is not None and count != last_count:
            last_count, last_change = count, time.time()
            set_status(dataset, "waiting_for_ingestion", f"{count}/{target} cases in {collection}", points=count)

        if count is not None and count > target:
            raise ChainFailed(f"{collection} holds {count} points but there are only {target} training "
                              "cases -- it was ingested more than once")
        if count == target:
            time.sleep(SETTLE_SECONDS)
            if qdrant_count(collection) == target:
                set_status(dataset, "ingestion_done", f"{target}/{target} cases in {collection}", points=target)
                return
            continue
        if time.time() - last_change > STALL_SECONDS:
            raise ChainFailed(f"ingestion into {collection} made no progress for {STALL_SECONDS // 60} min "
                              f"(stuck at {last_count}/{target}); the eval was not started")
        time.sleep(POLL_SECONDS)


def run_build(s: Dict) -> None:
    """Run the build script here until the collection holds every training case."""
    dataset, collection, target = s["dataset"], s["collection"], s["n_train"]

    # The build starts by resetting the collection, so make sure no other job is
    # in the middle of writing to it.
    set_status(dataset, "checking_collection_idle", f"watching {collection} for {IDLE_CHECK_SECONDS}s")
    before = qdrant_count(collection)
    time.sleep(IDLE_CHECK_SECONDS)
    after = qdrant_count(collection)
    if before != after:
        raise ChainFailed(f"{collection} changed from {before} to {after} points while idle-checking -- another "
                          "job is writing to it, so it was not reset. Use --wait for this dataset instead.")

    log_path = LOG_DIR / f"{dataset}_build.log"
    for attempt in range(1, BUILD_MAX_ATTEMPTS + 1):
        set_status(dataset, "building", f"attempt {attempt}/{BUILD_MAX_ATTEMPTS}, {target} cases to ingest",
                   attempt=attempt)
        code = run_child(dataset, s["build"], log_path, f"{dataset} build, attempt {attempt}")
        count = qdrant_count(collection)
        if code == 0 and count == target:
            set_status(dataset, "ingestion_done", f"{target}/{target} cases in {collection}", points=count)
            return
        log.warning("[%s] build attempt %s failed: exit code %s, %s/%s points", dataset, attempt, code, count, target)
        if attempt < BUILD_MAX_ATTEMPTS:
            set_status(dataset, "build_retry_wait",
                       f"attempt {attempt} failed (exit {code}, {count}/{target} points); retrying from scratch "
                       f"in {BUILD_RETRY_SECONDS // 60} min")
            time.sleep(BUILD_RETRY_SECONDS)
    raise ChainFailed(f"build failed {BUILD_MAX_ATTEMPTS} times (see {log_path.name}); the eval was not started")


def gate(s: Dict) -> None:
    """Refuse to evaluate against a case base whose entity extraction was broken."""
    dataset, collection, target = s["dataset"], s["collection"], s["n_train"]
    empty = qdrant_count(collection, only_empty_entities=True)
    if empty is None:
        raise ChainFailed(f"could not inspect {collection} in Qdrant")
    fraction = empty / target
    log.info("[%s] %s/%s cases (%.1f%%) have no entities (failed or genuinely entity-free)",
             dataset, empty, target, 100 * fraction)
    if fraction > MAX_EMPTY_ENTITY_FRACTION:
        raise ChainFailed(f"{empty}/{target} cases in {collection} have no entities ({100 * fraction:.0f}%); entity "
                          "extraction looks broken, so the eval was not started")
    with _status_lock:
        _status["chains"].setdefault(dataset, {})["cases_without_entities"] = empty


# ========== EVAL ==========
def _results_progress(results_dir: Path) -> int:
    """How many test questions have a saved result so far."""
    final = results_dir / RESULTS_NAME
    if final.exists():
        with open(final, "rb") as f:
            return len(pickle.load(f))
    steps = [int(p.stem.split("_ckp_")[-1]) for p in results_dir.glob(CHECKPOINT_GLOB)]
    return max(steps, default=0)


def _quota_wait_seconds(log_path: Path) -> Optional[int]:
    """If the last eval attempt stopped on an exhausted quota, how long until it is back.

    The Azure gateway answers with "Quota will be replenished in HH:MM:SS".
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    last_attempt = text[text.rfind("=" * 70):]
    match = re.search(r"replenished in (\d+):(\d+):(\d+)", last_attempt)
    if not match:
        return None
    hours, minutes, seconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + QUOTA_MARGIN_SECONDS


def run_eval(s: Dict) -> None:
    """Run the eval script until every test question has a result.

    The eval script stops early on a rate limit, saves a checkpoint, and then
    writes what it has as if it were the final result. A short result file is
    therefore set aside and the script is run again, which resumes from the
    checkpoint.
    """
    dataset, results_dir, target = s["dataset"], s["results_dir"], s["n_test"]
    log_path = LOG_DIR / f"{dataset}_eval.log"
    final = results_dir / RESULTS_NAME
    attempt, stuck, previous = 0, 0, -1

    while True:
        attempt += 1
        set_status(dataset, "evaluating", f"attempt {attempt}, {max(previous, 0)}/{target} questions done",
                   attempt=attempt)
        code = run_child(dataset, s["eval"], log_path, f"{dataset} eval, attempt {attempt}")
        done = _results_progress(results_dir)

        if code == 0 and done == target:
            break

        # Set aside a truncated "final" file so the next attempt resumes instead
        # of loading it from disk as a finished run. The eval script only
        # checkpoints every 100 questions and not when it stops on a rate limit,
        # so first keep the truncated file as a checkpoint: otherwise the answers
        # since the last multiple of 100 would be paid for again.
        if final.exists() and done < target:
            checkpoint = results_dir / CHECKPOINT_GLOB.replace("*", str(done))
            if done > 0 and not checkpoint.exists():
                shutil.copyfile(final, checkpoint)
            for name in (RESULTS_NAME, "metrics.json"):
                path = results_dir / name
                if path.exists():
                    os.replace(path, path.with_name(f"{path.stem}.partial-attempt{attempt}{path.suffix}"))

        stuck = 0 if done > previous else stuck + 1
        previous = done
        if stuck >= EVAL_MAX_STUCK_ATTEMPTS:
            raise ChainFailed(f"eval made no progress in {EVAL_MAX_STUCK_ATTEMPTS} attempts in a row "
                              f"(exit code {code}, {done}/{target} questions); see {log_path.name}")
        wait = _quota_wait_seconds(log_path) or EVAL_RETRY_SECONDS
        resume_at = datetime.fromtimestamp(time.time() + wait)
        set_status(dataset, "eval_retry_wait",
                   f"attempt {attempt} stopped at {done}/{target} (exit {code}); resuming at "
                   f"{resume_at:%Y-%m-%d %H:%M}" + (" when the API quota is replenished" if wait != EVAL_RETRY_SECONDS else ""))
        time.sleep(wait)

    with open(final, "rb") as f:
        results = pickle.load(f)
    errors = sum(1 for r in results if "error" in r)
    metrics_path = results_dir / "metrics.json"
    detail = f"{target}/{target} questions, {errors} failed with an error; metrics in {metrics_path.relative_to(ROOT)}"
    if not metrics_path.exists():
        raise ChainFailed(f"eval finished but {metrics_path.name} is missing; see {log_path.name}")
    if errors > MAX_ERROR_ROW_FRACTION * target:
        set_status(dataset, "done_with_errors", detail + " -- too many errors to trust the metrics", error_rows=errors)
    else:
        set_status(dataset, "done", detail, error_rows=errors)


def chain(s: Dict, mode: str) -> None:
    dataset = s["dataset"]
    try:
        if mode == "wait":
            wait_for_external_ingestion(s)
        else:
            run_build(s)
        gate(s)
        run_eval(s)
    except ChainFailed as e:
        set_status(dataset, "failed", str(e))
    except Exception:
        log.error("[%s] unexpected error\n%s", dataset, traceback.format_exc())
        set_status(dataset, "failed", "unexpected error in the runner; see orchestrator.log")


# ========== PREFLIGHT ==========
def preflight(plan: Dict[str, str], allow_existing_results: bool) -> List[str]:
    """Everything that can be verified before committing hours of API calls."""
    problems: List[str] = []
    if not PYTHON.exists():
        problems.append(f"project venv not found: {PYTHON}")

    for dataset, mode in plan.items():
        try:
            s = resolve(dataset)
        except Exception as e:
            problems.append(f"[{dataset}] {e}")
            continue

        print(f"[{dataset}] ingestion: {mode:5s}  collection: {s['collection']}  "
              f"({qdrant_count(s['collection'])}/{s['n_train']} cases)")
        print(f"[{dataset}] build: {s['build_generator']} / {s['build_model']}   "
              f"eval: {'Azure' if s['eval_use_azure'] == 'True' else 'OpenAI'} / {s['eval_model']}   "
              f"{s['n_test']} test questions -> {s['results_dir'].relative_to(ROOT)}")

        if s["collection"] != s["eval_collection"]:
            problems.append(f"[{dataset}] build writes {s['collection']} but eval reads {s['eval_collection']}")
        if (s["build_model"], s["build_generator"]) != (BUILD_MODEL, BUILD_GENERATOR):
            problems.append(f"[{dataset}] build must use {BUILD_GENERATOR} / {BUILD_MODEL}, "
                            f"script has {s['build_generator']} / {s['build_model']}")
        if (s["eval_model"], s["eval_use_azure"]) != s["eval_on"]:
            problems.append(f"[{dataset}] eval must use MODEL={s['eval_on'][0]} / USE_AZURE={s['eval_on'][1]}, "
                            f"script has MODEL={s['eval_model']} / USE_AZURE={s['eval_use_azure']}")
        if s["eval_pipelines"] not in ('["CBR-CDB"]', "['CBR-CDB']"):
            problems.append(f"[{dataset}] eval must have EVALUATE = [\"CBR-CDB\"], script has {s['eval_pipelines']}")
        if mode == "wait" and qdrant_count(s["collection"]) is None:
            problems.append(f"[{dataset}] --wait, but collection {s['collection']} does not exist "
                            "(or Qdrant is unreachable)")
        if s["results_dir"].exists() and any(s["results_dir"].glob("*.pkl")) and not allow_existing_results:
            problems.append(f"[{dataset}] {s['results_dir'].relative_to(ROOT)} already holds result files; the eval "
                            "would resume from or reuse them. Move them away, or pass --allow-existing-results "
                            "if they belong to this same run.")

    if qdrant_count("__preflight__") is None:
        try:
            urllib.request.urlopen(f"{QDRANT_URL}/collections", timeout=10).read()
        except (urllib.error.URLError, OSError) as e:
            problems.append(f"Qdrant is not reachable at {QDRANT_URL}: {e}")
    return problems


def ping() -> int:
    """One live call per provider/model pair the chains depend on."""
    os.chdir(ROOT)
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    from src.configs import LLMConfig
    from src.generator import AzureAIAgent, OpenAIAgent

    failed = 0
    pairs = [("build generator", OpenAIAgent, BUILD_MODEL), ("build fallback", AzureAIAgent, BUILD_MODEL)]
    for dataset, cfg in DATASETS.items():
        model, use_azure = cfg["eval_on"]
        pairs.append((f"{dataset} eval", AzureAIAgent if use_azure == "True" else OpenAIAgent, model))
        pairs.append((f"{dataset} fallback", OpenAIAgent, model))
    for role, agent, model in pairs:
        try:
            reply = agent(LLMConfig(model=model)).generate([("user", "Reply with the single word OK.")])
            print(f"ok    {role:16s} {agent.__name__:13s} {model:13s} -> {reply.strip()[:20]!r}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {role:16s} {agent.__name__:13s} {model:13s} -> {str(e)[:200]}")
    return failed


# ========== PROCESS CONTROL ==========
def _pid_alive(pid: int) -> bool:
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(handle)
    return code.value == 259  # STILL_ACTIVE


def _running_pid() -> Optional[int]:
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def run(plan: Dict[str, str], allow_existing_results: bool, sequential: bool = False) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:  # started with pythonw: no console to write to
        sys.stdout = sys.stderr = open(LOG_DIR / "orchestrator.out", "a", encoding="utf-8", buffering=1)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_DIR / "orchestrator.log", encoding="utf-8")],
    )

    if _running_pid():
        log.error("another runner is already active (pid %s); not starting a second one", _running_pid())
        return 2
    PID_FILE.write_text(str(os.getpid()))
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))

    log.info("=" * 60)
    log.info("runner started (pid %s), plan: %s%s", os.getpid(), plan, ", sequential" if sequential else "")
    problems = preflight(plan, allow_existing_results)
    if problems:
        for problem in problems:
            log.error("preflight: %s", problem)
        return 1

    # Keep the machine from idle-sleeping while the chains run (cleared on exit).
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    _status.update(pid=os.getpid(), started=datetime.now().isoformat(timespec="seconds"), plan=plan)
    threads = []
    for dataset, mode in plan.items():
        set_status(dataset, "queued" if sequential else "starting", f"ingestion mode: {mode}", ingestion=mode)
    for dataset, mode in plan.items():
        settings = resolve(dataset)
        if sequential:
            # One chain at a time, in the order given; a failed chain does not
            # stop the ones after it.
            chain(settings, mode)
            continue
        thread = threading.Thread(target=chain, args=(settings, mode), name=dataset)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    log.info("runner finished: %s", {d: c.get("state") for d, c in _status["chains"].items()})
    return 0 if all(c.get("state") == "done" for c in _status["chains"].values()) else 1


def show_status() -> int:
    pid = _running_pid()
    print(f"runner: {'running, pid ' + str(pid) if pid else 'not running'}")
    if STATUS_FILE.exists():
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        for dataset, entry in status.get("chains", {}).items():
            print(f"[{dataset}] {entry.get('state')}  ({entry.get('updated')})\n    {entry.get('detail', '')}")
    for dataset in DATASETS:
        s = resolve(dataset)
        print(f"[{dataset}] {s['collection']}: {qdrant_count(s['collection'])}/{s['n_train']} cases, "
              f"results: {_results_progress(s['results_dir']) if s['results_dir'].exists() else 0}/{s['n_test']}")
    return 0


def stop() -> int:
    pid = _running_pid()
    if not pid:
        print("runner is not running")
        return 0
    # /T also ends the build/eval child it is waiting on
    return subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--wait", nargs="*", default=[], choices=list(DATASETS), metavar="DATASET",
                       help="ingestion is already running elsewhere; wait for it")
        p.add_argument("--build", nargs="*", default=[], choices=list(DATASETS), metavar="DATASET",
                       help="run the ingestion here")
        p.add_argument("--allow-existing-results", action="store_true")
        p.add_argument("--sequential", action="store_true",
                       help="run the chains one after another, in the order given, instead of in parallel")
    sub.add_parser("ping")
    sub.add_parser("status")
    sub.add_parser("stop")
    sub.add_parser("exec").add_argument("script")
    args = parser.parse_args()

    if args.command == "exec":
        exec_script(args.script)
        return 0
    if args.command == "ping":
        return ping()
    if args.command == "status":
        return show_status()
    if args.command == "stop":
        return stop()

    if set(args.wait) & set(args.build):
        parser.error("a dataset cannot be both --wait and --build")
    # Insertion order is the run order under --sequential
    plan = {**{d: "wait" for d in args.wait}, **{d: "build" for d in args.build}}
    if not plan:
        parser.error("nothing to do: pass --wait and/or --build")

    if args.command == "check":
        problems = preflight(plan, args.allow_existing_results)
        for problem in problems:
            print(f"PROBLEM: {problem}")
        print("preflight ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    return run(plan, args.allow_existing_results, args.sequential)


if __name__ == "__main__":
    sys.exit(main())
