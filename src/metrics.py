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



import sqlglot
from sqlglot import exp

OP_MAP = {
    exp.EQ: "=", exp.NEQ: "!=", exp.GT: ">", 
    exp.GTE: ">=", exp.LT: "<", exp.LTE: "<=",
}

def parse_sql_(sql, headerDic, tableDic):
    def parse_query(node):
        if not isinstance(node, exp.Select):
            return None
            
        form = {"sel": "", "agg": [], "tab": [], "cond": [], "from": []}
        
        # SELECT
        select_exprs = node.args.get("expressions", [])
        for expr in select_exprs:
            if isinstance(expr, exp.Count):
                form["sel"] = "count"
                for col in expr.find_all(exp.Column):
                    col_name = col.sql().lower()
                    if col_name in headerDic:
                        form["agg"].append(col_name)
            elif isinstance(expr, exp.Min):
                form["sel"] = "min"
                for col in expr.find_all(exp.Column):
                    col_name = col.sql().lower()
                    if col_name in headerDic:
                        form["agg"].append(col_name)
            elif isinstance(expr, exp.Max):
                form["sel"] = "max"
                for col in expr.find_all(exp.Column):
                    col_name = col.sql().lower()
                    if col_name in headerDic:
                        form["agg"].append(col_name)
            elif isinstance(expr, exp.Avg):
                form["sel"] = "avg"
                for col in expr.find_all(exp.Column):
                    col_name = col.sql().lower()
                    if col_name in headerDic:
                        form["agg"].append(col_name)
            else:
                for col in expr.find_all(exp.Column):
                    col_name = col.sql().lower()
                    if col_name in headerDic:
                        form["agg"].append(col_name)
        
        # FROM
        from_ = node.args.get("from_")
        if from_:
            table_expr = from_.this
            if isinstance(table_expr, exp.Table):
                table_id = table_expr.this
                if isinstance(table_id, exp.Identifier):
                    tname = str(table_id.this).lower()
                    if tname in tableDic:
                        form["tab"].append(tname)
            elif isinstance(table_expr, exp.Subquery):
                parsed = parse_query(table_expr.this)
                if parsed:
                    form["from"].append(parsed)
        
        # WHERE
        where = node.args.get("where")
        if where:
            # Extract subqueries
            for sub in where.find_all(exp.Subquery):
                parent = sub.parent
                is_direct = True
                while parent and parent != where:
                    if isinstance(parent, exp.Subquery):
                        is_direct = False
                        break
                    parent = parent.parent
                
                if is_direct:
                    parsed = parse_query(sub.this)
                    if parsed:
                        form["from"].append(parsed)
            
            # Extract predicates - need to handle Not wrapper
            for node_item in where.find_all(exp.Predicate):
                parent = node_item.parent
                in_subquery = False
                while parent and parent != where:
                    if isinstance(parent, exp.Subquery):
                        in_subquery = True
                        break
                    parent = parent.parent
                
                if in_subquery:
                    continue
                
                # Check if wrapped in Not
                is_negated = isinstance(node_item.parent, exp.Not)
                    
                if isinstance(node_item, exp.Between):
                    form["cond"].append([node_item.this.sql().lower(), "between", 
                        f"{node_item.args['low'].sql().lower()}|{node_item.args['high'].sql().lower()}"])
                elif isinstance(node_item, exp.Is):
                    op = "is not" if is_negated else "is"
                    form["cond"].append([node_item.this.sql().lower(), op, node_item.expression.sql().lower()])
                elif isinstance(node_item, exp.In):
                    # IN condition - subquery already handled above
                    pass
                else:
                    for cls, op in OP_MAP.items():
                        if isinstance(node_item, cls):
                            form["cond"].append([node_item.left.sql().lower(), op, node_item.right.sql().lower()])
                            break
        
        form["agg"] = sorted(list(set(form["agg"])))
        form["tab"] = sorted(list(set(form["tab"])))
        form["cond"] = sorted(form["cond"])
        return form
    
    return parse_query(sqlglot.parse_one(sql))


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