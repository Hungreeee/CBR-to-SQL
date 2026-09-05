import re
from typing import List, Dict
from collections import Counter

from src.utils import *


def execution_accuracy(results_dataset: List[Dict], sql_db: query):
    exact_count = unordered_count = 0
    
    for i, item in enumerate(results_dataset):
        gold_sql = str(item["gold_sql"]).replace("%y", "%Y")
        pred_sql = str(item["predicted_sql"]).replace("%y", "%Y")
        
        # Handle impossible queries
        if gold_sql == "null":
            if pred_sql == "None":
                exact_count += 1
                unordered_count += 1
            else:
                print(
                    f"\n{item['question']}\n"
                    f"PRED: {pred_sql}\n"
                    f"GOLD: {gold_sql}\n"
                )
            continue
        
        # Execute queries
        try:
            gold_cur = sql_db.execute_sql(gold_sql)
            gold_out = gold_cur.fetchall()
            pred_cur = sql_db.execute_sql(pred_sql)
            pred_out = pred_cur.fetchall()
        except Exception:
            print(
                f"\n{item['question']}\n"
                f"PRED: {pred_sql}\n"
                f"GOLD: {gold_sql}\n"
            )
            continue
        
        # Check matches
        exact_match = pred_out == gold_out
        unordered_match = unordered_match = (
            Counter(tuple(sorted(row, key=str)) for row in gold_out)
            ==
            Counter(tuple(sorted(row, key=str)) for row in pred_out)
        )

        
        exact_count += exact_match
        unordered_count += unordered_match
        
        if not unordered_match:
            print(
                f"\n{item['question']}\n"
                f"PRED: {pred_sql}\n"
                f"GOLD: {gold_sql}\n"
                f"PRED_OUT: {Counter(tuple(sorted(row, key=str)) for row in pred_out)}\n"
                f"GOLD_OUT: {Counter(tuple(sorted(row, key=str)) for row in gold_out)}\n"
            )
    
    return {
        "exact_execution_accuracy": exact_count / len(results_dataset),
        "standard_execution_accuracy": unordered_count / len(results_dataset)
    }


def normalize_row(results):
    return [tuple(sorted(row, key=str)) for row in results]


def logic_form_accuracy_(cbr_dataset: List[Dict], rag_dataset: List[Dict], db_model: query):
    db_head = db_model.db_head

    headerDic = []
    for tb in db_head:
        for hd in db_head[tb]:
            headerDic.append('.'.join([tb, hd]).lower())

    tableDic = []
    for tb in db_head:
        tableDic.append(tb.lower())

    def parse_results(result_dataset):
        outGen = []
        outTtt = []
        for line in result_dataset:
            gold_sql = line['gold_sql']
            pred_sql = line['predicted_sql']

            if pred_sql is None:
                pred_sql = "None"

            if gold_sql == "null":
                if pred_sql == "None":
                    outGen.append(None)
                    outTtt.append(None)
                else:
                    outGen.append(None)
                    outTtt.append("null")
                continue

            if pred_sql.strip() == '' or pred_sql == 'None':
                pred_sql = 'SELECT TEST."NOTHING" FROM TEST WHERE TEST."NOTHING" = "NONE"'

            gen = re.split('<stop>', pred_sql)[0]
            sqlG = parse_sql(gen, headerDic, tableDic)
            outGen.append(sqlG)

            sqlT = parse_sql(gold_sql, headerDic, tableDic)
            outTtt.append(sqlT)

        return outGen, outTtt

    cbr_gen, cbr_truth = parse_results(cbr_dataset)
    rag_gen, rag_truth = parse_results(rag_dataset)

    lf_count = {
        "total": 0,
        "agg_op": 0,
        "agg_col": 0,
        "table": 0,
        "condition_column_operation": 0,
        "condition_value": 0,
    }

    for k in range(len(cbr_gen)):
        cbr_lf = cbr_gen[k]
        rag_lf = rag_gen[k]
        gold_lf = cbr_truth[k]  # gold is the same for both

        cbr_correct = (cbr_lf is None and gold_lf is None) or (cbr_lf == gold_lf)
        rag_correct = (rag_lf is None and gold_lf is None) or (rag_lf == gold_lf)

        # Print cases where RAG succeeds but CBR fails
        if rag_correct and not cbr_correct:
            print("=" * 40)
            print(f"QUESTION: {cbr_dataset[k]['question']}")
            print("-" * 40)
            print("CBR (FAIL):")
            print(cbr_dataset[k]['predicted_sql'])
            print(cbr_lf)
            print("-" * 40)
            print("RAG (SUCCESS):")
            print(rag_dataset[k]['predicted_sql'])
            print(rag_lf)
            print("-" * 40)
            print("GOLD:")
            print(cbr_dataset[k]['gold_sql'])
            print(gold_lf)
            print()

        # LF breakdown is computed on CBR predictions (vs gold)
        if cbr_lf is None and gold_lf is None:
            lf_count["total"] += 1
            continue

        if cbr_lf is None or gold_lf is None:
            continue

        if cbr_lf == gold_lf:
            lf_count["total"] += 1

        if cbr_lf['sel'] == gold_lf['sel']:
            lf_count["agg_op"] += 1

        if cbr_lf['agg'] == gold_lf['agg']:
            lf_count["agg_col"] += 1

        if cbr_lf['tab'] == gold_lf['tab']:
            lf_count["table"] += 1

        arrG = [wd[:2] for wd in cbr_lf['cond']]
        arrT = [wd[:2] for wd in gold_lf['cond']]
        if arrG == arrT:
            lf_count["condition_column_operation"] += 1

        arrG = [wd[:3] for wd in cbr_lf['cond']]
        arrT = [wd[:3] for wd in gold_lf['cond']]
        if arrG == arrT:
            lf_count["condition_value"] += 1

    return {cat: (cnt / len(cbr_gen)) for (cat, cnt) in lf_count.items()}


def logic_form_accuracy(result_dataset: List[Dict], db_model: query):
    db_head = db_model.db_head

    headerDic = []
    for tb in db_head:
        for hd in db_head[tb]:
            headerDic.append('.'.join([tb, hd]).lower())

    tableDic = []
    for tb in db_head:
        tableDic.append(tb.lower())

    outGen = []
    outTtt = []

    for line in result_dataset:
        gold_sql = line['gold_sql']
        pred_sql = line['predicted_sql']

        if pred_sql is None:
            pred_sql = "None"
        
        # Handle impossible queries
        if gold_sql == "null":
            if pred_sql == "None":
                # both impossible → correct
                outGen.append(None)
                outTtt.append(None)
            else:
                # predicted SQL when it should be impossible → wrong
                outGen.append(None)
                outTtt.append("null")   # <-- key change
            continue
        
        # Handle empty predictions
        if pred_sql.strip() == '' or pred_sql == 'None':
            pred_sql = 'SELECT TEST."NOTHING" FROM TEST WHERE TEST."NOTHING" = "NONE"'
        
        gen = re.split('<stop>', pred_sql)[0]
        sqlG = parse_sql(gen, headerDic, tableDic)
        outGen.append(sqlG)
        
        sqlT = parse_sql(gold_sql, headerDic, tableDic)
        outTtt.append(sqlT)

    lf_count = {
        "total": 0,
        "agg_op": 0,
        "agg_col": 0,
        "table": 0,
        "condition_column_operation": 0,
        "condition_value": 0,
    }

    cnt = 0

    for k in range(len(outGen)):
        gen_lf = outGen[k]
        gold_lf = outTtt[k]

        # both None → correct
        if gen_lf is None and gold_lf is None:
            lf_count["total"] += 1
            continue

        # one None → wrong
        if gen_lf is None or gold_lf is None:
            continue

        # if results correct
        if gen_lf == gold_lf:
            lf_count["total"] += 1
        # if results incorrect
        else:
            if result_dataset[k]["predicted_sql"] != result_dataset[k]["gold_sql"]:
                print(result_dataset[k]["question"])
                print("-"*10)
                print("PREDS:")
                print(result_dataset[k]["predicted_sql"])                
                print(gen_lf)
                print("-"*10)
                print("TRUTH:")
                print(result_dataset[k]["gold_sql"])
                print(gold_lf)
                print("-"*10)
                print()
            else:
                print("ERRORS")

        if gen_lf['sel'] == gold_lf['sel']:
            lf_count["agg_op"] += 1 

        if gen_lf['agg'] == gold_lf['agg']:
            lf_count["agg_col"] += 1

        if gen_lf['tab'] == gold_lf['tab']:
            lf_count["table"] += 1 

        arrG = [wd[:2] for wd in gen_lf['cond']]
        arrT = [wd[:2] for wd in gold_lf['cond']]
        if arrG == arrT:
            lf_count["condition_column_operation"] += 1

        arrG = [wd[:3] for wd in gen_lf['cond']]
        arrT = [wd[:3] for wd in gold_lf['cond']]
        if arrG == arrT:
            lf_count["condition_value"] += 1 

    return {cat: (cnt / len(outGen)) for (cat, cnt) in lf_count.items()}


def parse_sql(sql, headerDic, tableDic):
    sqlForm = {}
    
    arr = re.split('where', sql.lower())
    qlead = re.split('from', arr[0])
    qlead[0] = ",".join([i.strip() for i in qlead[0].split(",")])
    qagg = re.split('\s', qlead[0])
    qagg = list(filter(None, qagg))

    normalized = []
    for token in qagg:
        parts = re.findall(r'[^\(\),]+|[\(\),]', token)
        normalized.extend(part for part in parts if part.strip())
    qagg = normalized

    if len(qagg) > 1: 
        if qagg[1] == 'count' or qagg[1] == 'min' or qagg[1] == 'max' or qagg[1] == 'avg':
            sqlForm['sel'] = qagg[1]
        else:
            sqlForm['sel'] = ''
    else:
        sqlForm['sel'] = ''

    itm = []
    for wd in qagg:
        if wd in headerDic:
            itm.append(wd)
    sqlForm['agg'] = itm
    
    itm = []
    qtab = re.split('\s', qlead[1])
    qtab = list(filter(None, qtab))
    for wd in qtab:
        if wd in tableDic:
            itm.append(wd)
    sqlForm['tab'] = itm
        
    qtail = re.split('and', arr[-1])
    itm = []
    for cond in qtail:
        cond = re.split('\s', cond)
        cond = list(filter(None, cond))
        if len(cond) > 2:
            condVal = ' '.join(cond[2:])
            condVal = re.split('\"|\s|\'', condVal)
            condVal = ' '.join(list(filter(None, condVal)))
            itm.append(cond[:2] + [condVal])
    sqlForm['cond'] = sorted(itm)
    
    return sqlForm


def normalize_val(v):
    """Lowercase, strip quotes/spaces, collapse whitespace — mirror parse_sql's value handling."""
    if v is None:
        return ""
    v = re.split(r'\"|\s|\'', str(v))
    return ' '.join(filter(None, v)).lower()

def _clean(s):
    return (s or '').replace('"', '').replace("'", '').strip().lower()

def gold_entities_from_sql(gold_lf):
    gold = []
    for cond in gold_lf['cond']:
        if len(cond) < 3:
            continue
        col_full = _clean(cond[0])          # 'demographic."diagnosis"' -> 'demographic.diagnosis'
        val = normalize_val(cond[2])
        if '.' in col_full:
            tab, col = col_full.split('.', 1)
        else:
            tab, col = '', col_full
        gold.append((tab.strip(), col.strip(), val))
    return gold

def pred_entities_from_lookup(entity_list):
    preds = []
    for e in entity_list:
        if not e.get('match_found', False):
            continue
        tab = _clean(e.get('table'))
        col = _clean(e.get('column'))       # '"DIAGNOSIS"' -> 'diagnosis'
        val = normalize_val(e.get('best_match'))
        preds.append((tab, col, val))
    return preds


def entity_resolution_accuracy(result_dataset, db_model,
                               entities_field='entities',
                               covered_columns=None):
    """
    covered_columns: set of (table, column) lowercased pairs that the lookup
    table actually indexes. Gold conditions on columns OUTSIDE this set are
    excluded, since the lookup is by design not responsible for them
    (numeric/ID/temporal fields filtered out during construction).
    If None, falls back to the set of (table,column) the lookup ever predicts.
    """
    db_head = db_model.db_head
    headerDic, tableDic = [], []
    for tb in db_head:
        for hd in db_head[tb]:
            headerDic.append('.'.join([tb, hd]).lower())
        tableDic.append(tb.lower())

    # Fallback: infer covered columns from predictions
    if covered_columns is None:
        covered_columns = set()
        for line in result_dataset:
            for e in line.get(entities_field, []):
                if e.get('match_found', False):
                    covered_columns.add((_clean(e.get('table')),
                                         _clean(e.get('column'))))

    q_total = q_t = q_c = q_v = 0
    e_total = e_t = e_c = e_v = 0

    for line in result_dataset:
        if line['gold_sql'] == 'null':
            continue
        gold_lf = parse_sql(line['gold_sql'], headerDic, tableDic)
        gold = gold_entities_from_sql(gold_lf)
        # keep only conditions on columns the lookup is responsible for
        gold = [(t, c, v) for (t, c, v) in gold if (t, c) in covered_columns]
        if not gold:
            continue

        preds = pred_entities_from_lookup(line.get(entities_field, []))
        pred_tables = {p[0] for p in preds}
        pred_cols   = {(p[0], p[1]) for p in preds}
        pred_full   = {(p[0], p[1], p[2]) for p in preds}

        q_total += 1
        all_t = all_c = all_v = True
        for (gt, gc, gv) in gold:
            e_total += 1
            t_ok = gt in pred_tables
            c_ok = (gt, gc) in pred_cols
            v_ok = (gt, gc, gv) in pred_full
            e_t += t_ok; e_c += c_ok; e_v += v_ok
            all_t &= t_ok; all_c &= c_ok; all_v &= v_ok
        q_t += all_t; q_c += all_c; q_v += all_v

    return {
        'per_query': {'table': q_t/q_total, 'table_column': q_c/q_total,
                      'table_column_val': q_v/q_total, 'n_queries': q_total},
        'per_entity': {'table': e_t/e_total, 'table_column': e_c/e_total,
                       'table_column_val': e_v/e_total, 'n_entities': e_total},
    }


def compute_reliability_score(
    results_dataset: List[Dict],
    sql_db: query,  # Your database connection/executor object
    penalty_c: int = 10
) -> Dict[str, float]:
    """
    Compute Reliability Score (RS) for text-to-SQL models.
    
    Args:
        results_dataset: List of dictionaries, each containing:
            - 'question': str
            - 'gold_sql': str (ground truth SQL, "null" for unanswerable)
            - 'predicted_sql': str (model's prediction, "None" for abstention)
        sql_db: Database connection with execute_sql method
        penalty_c: Penalty parameter (c) for incorrect predictions
    
    Returns:
        Dictionary with RS scores and detailed statistics
    """
    
    total_samples = len(results_dataset)
    sample_scores = []
    
    # Statistics
    stats = {
        'answerable_correct': 0,
        'answerable_abstain': 0,
        'answerable_incorrect': 0,
        'unanswerable_abstain': 0,
        'unanswerable_incorrect': 0,
        'total_answerable': 0,
        'total_unanswerable': 0,
    }
    
    for i, item in enumerate(results_dataset):
        gold_sql = str(item["gold_sql"]).replace("%y", "%Y")
        pred_sql = str(item["predicted_sql"]).replace("%y", "%Y")
        
        # Determine if question is answerable
        is_answerable = (gold_sql != "null")
        
        if is_answerable:
            stats['total_answerable'] += 1
            
            # Check if model abstained
            if pred_sql == "None":
                # Case 2: Answerable question, model abstains
                sample_score = 0
                stats['answerable_abstain'] += 1
            else:
                # Model attempted to answer - check execution accuracy
                try:
                    # Execute both queries
                    gold_cur = sql_db.execute_sql(gold_sql)
                    gold_out = gold_cur.fetchall()
                    pred_cur = sql_db.execute_sql(pred_sql)
                    pred_out = pred_cur.fetchall()
                    
                    # Check for unordered match (standard execution accuracy)
                    gold_counter = Counter(
                        tuple(sorted(row, key=str)) for row in gold_out
                    )
                    pred_counter = Counter(
                        tuple(sorted(row, key=str)) for row in pred_out
                    )
                    
                    is_correct = (gold_counter == pred_counter)
                    
                    if is_correct:
                        # Case 1: Answerable question, correct SQL
                        sample_score = 1
                        stats['answerable_correct'] += 1
                    else:
                        # Case 3: Answerable question, incorrect SQL
                        sample_score = -penalty_c
                        stats['answerable_incorrect'] += 1
                        
                except Exception as e:
                    # If execution fails, treat as incorrect SQL
                    sample_score = -penalty_c
                    stats['answerable_incorrect'] += 1
                    
        else:  # Unanswerable question
            stats['total_unanswerable'] += 1
            
            if pred_sql == "None":
                # Case 5: Unanswerable question, model abstains
                sample_score = 1
                stats['unanswerable_abstain'] += 1
            else:
                # Case 4: Unanswerable question, model generates SQL
                sample_score = -penalty_c
                stats['unanswerable_incorrect'] += 1
        
        sample_scores.append(sample_score)
    
    # Compute average RS as percentage
    avg_score = sum(sample_scores) / total_samples * 100
    
    # Compute RS for different penalty values if needed
    rs_results = {}
    if penalty_c == 10:
        rs_results['RS(10)'] = avg_score
    elif penalty_c == 0:
        rs_results['RS(0)'] = avg_score
    elif penalty_c == total_samples:
        rs_results['RS(N)'] = avg_score
    
    # Compute additional metrics
    if stats['total_answerable'] > 0:
        answerable_accuracy = stats['answerable_correct'] / stats['total_answerable'] * 100
        answerable_abstention_rate = stats['answerable_abstain'] / stats['total_answerable'] * 100
        answerable_error_rate = stats['answerable_incorrect'] / stats['total_answerable'] * 100
    else:
        answerable_accuracy = answerable_abstention_rate = answerable_error_rate = 0
    
    if stats['total_unanswerable'] > 0:
        unanswerable_detection_rate = stats['unanswerable_abstain'] / stats['total_unanswerable'] * 100
        unanswerable_error_rate = stats['unanswerable_incorrect'] / stats['total_unanswerable'] * 100
    else:
        unanswerable_detection_rate = unanswerable_error_rate = 0
    
    # Compile all results
    results = {
        'reliability_score': avg_score,
        'penalty_used': penalty_c,
        'sample_scores': sample_scores,
        'stats': stats,
        'detailed_metrics': {
            'answerable_accuracy': answerable_accuracy,
            'answerable_abstention_rate': answerable_abstention_rate,
            'answerable_error_rate': answerable_error_rate,
            'unanswerable_detection_rate': unanswerable_detection_rate,
            'unanswerable_error_rate': unanswerable_error_rate,
        }
    }
    
    # Add RS variants if computing multiple
    rs_results.update(results)
    
    return rs_results


def compute_rs_variants(
    results_dataset: List[Dict],
    sql_db,
    compute_all: bool = True
) -> Dict[str, float]:
    """
    Compute RS for different penalty values.
    
    Args:
        results_dataset: Same as compute_reliability_score
        sql_db: Database connection
        compute_all: If True, compute RS(0), RS(10), and RS(N)
    
    Returns:
        Dictionary with RS scores for different penalties
    """
    n = len(results_dataset)
    results = {}
    
    if compute_all:
        # Compute RS(0) - no penalty
        rs0 = compute_reliability_score(results_dataset, sql_db, penalty_c=0)
        results['RS(0)'] = rs0['reliability_score']
        
        # Compute RS(10) - moderate penalty (main metric)
        rs10 = compute_reliability_score(results_dataset, sql_db, penalty_c=10)
        results['RS(10)'] = rs10['reliability_score']
        
        # Compute RS(N) - severe penalty
        rsN = compute_reliability_score(results_dataset, sql_db, penalty_c=n)
        results['RS(N)'] = rsN['reliability_score']
        
        # Add detailed stats from RS(10) since it's the main metric
        results['detailed_stats'] = rs10['stats']
        results['detailed_metrics'] = rs10['detailed_metrics']
    else:
        # Just compute RS(10)
        rs10 = compute_reliability_score(results_dataset, sql_db, penalty_c=10)
        results['RS(10)'] = rs10['reliability_score']
        results['detailed_stats'] = rs10['stats']
        results['detailed_metrics'] = rs10['detailed_metrics']
    
    return results