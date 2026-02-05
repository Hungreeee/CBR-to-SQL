import re
from typing import List, Dict, Optional
from collections import Counter
from src.utils import *


CURRENT_DATE = "2100-12-31"
CURRENT_TIME = "23:59:00"
NOW = f"{CURRENT_DATE} {CURRENT_TIME}"
PRECOMPUTED_DICT = {
    'temperature': (35.5, 38.1),
    'sao2': (95.0, 100.0),
    'heart rate': (60.0, 100.0),
    'respiration': (12.0, 18.0),
    'systolic bp': (90.0, 120.0),
    'diastolic bp': (60.0, 90.0),
    'mean bp': (60.0, 110.0)
}

TIME_PATTERN = r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+ INTERVAL (\d+) (MONTH|YEAR|DAY)\)"

def convert_date_function(match):
    function = match.group(1)
    date = match.group(2)
    number = match.group(3)
    unit = match.group(4).lower()
    
    # Use singular form when number is 1
    if number == '1':
        unit = unit.rstrip('s')
    else:
        unit += 's' if not unit.endswith('s') else ''
    
    # Determine the sign based on the function (DATE_SUB or DATE_ADD)
    sign = '-' if function == 'DATE_SUB' else '+'
    
    return f"datetime({date}, '{sign}{number} {unit}')"

def post_process_sql(query: str | None) -> str:
    if query is None:
        return "None"

    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')

    # Convert MySQL to SQLite functions
    query = re.sub(TIME_PATTERN, convert_date_function, query)

    if "current_time" in query: # strftime('%J',current_time) => strftime('%J','2100-12-31 23:59:00')
        query = query.replace("current_time", f"'{NOW}'")
    if "current_date" in query: # strftime('%J',current_date) => strftime('%J','2100-12-31')
        query = query.replace("current_date", f"'{CURRENT_DATE}'")
    if "'now'" in query: # 'now' => '2100-12-31 23:59:00'
        query = query.replace("'now'", f"'{NOW}'")
    if "NOW()" in query: # NOW() => '2100-12-31 23:59:00'
        query = query.replace("NOW()", f"'{NOW}'")
    if "CURDATE()" in query: # CURDATE() => '2100-12-31'
        query = query.replace("CURDATE()", f"'{CURRENT_DATE}'")
    if "CURTIME()" in query: # CURTIME() => '23:59:00'
        query = query.replace("CURTIME()", f"'{CURRENT_TIME}'")
        
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list)==1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in PRECOMPUTED_DICT:
                vital_range = PRECOMPUTED_DICT[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")

    query = query.replace("%y", "%Y").replace('%j', '%J')
    query = query.replace("\"None\"", "None")
    return query


def execution_accuracy(results_dataset: List[Dict], sql_db: query):
    exact_count = unordered_count = 0
    
    for i, item in enumerate(results_dataset):
        gold_sql = post_process_sql(str(item["gold_sql"]))
        pred_sql = post_process_sql(str(item["predicted_sql"]))
        
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


def parse_sql_nested(sql: str, headerDic: List[str], tableDic: List[str]) -> Optional[Dict]:
    """
    Parse SQL with nested subquery support.
    Keeps the hierarchical structure of subqueries.
    """
    if not sql or sql.strip() == '' or sql.lower() == 'None':
        return None
    
    try:
        sql = sql.lower().strip()
        
        # Extract and parse subqueries first
        subqueries = []
        sql_processed = extract_subqueries(sql, headerDic, tableDic, subqueries)
        
        # Parse the main query
        sqlForm = parse_single_query(sql_processed, headerDic, tableDic)
        
        # Add subquery information (keep hierarchical structure)
        sqlForm['subqueries'] = subqueries
        
        return sqlForm
        
    except Exception as e:
        print(f"Parse error for SQL: {sql[:100]}... Error: {e}")
        return None


def extract_subqueries(sql: str, headerDic: List[str], tableDic: List[str], 
                       subqueries: List[Dict]) -> str:
    """
    Extract subqueries and replace them with placeholders.
    Returns modified SQL with placeholders.
    """
    
    def find_matching_paren(text, start_pos):
        """Find the matching closing parenthesis."""
        count = 1
        pos = start_pos + 1
        while pos < len(text) and count > 0:
            if text[pos] == '(':
                count += 1
            elif text[pos] == ')':
                count -= 1
            pos += 1
        return pos if count == 0 else -1
    
    # Keep extracting subqueries until none remain
    modified_sql = sql
    pos = 0
    
    while pos < len(modified_sql):
        # Look for "( SELECT" pattern (case insensitive)
        match = re.search(r'\(\s*select\s+', modified_sql[pos:], re.IGNORECASE)
        
        if not match:
            break
        
        # Get the actual position in the full string
        start_pos = pos + match.start()
        
        # Find matching closing parenthesis
        end_pos = find_matching_paren(modified_sql, start_pos)
        
        if end_pos == -1:
            # No matching paren found, skip this one
            pos = start_pos + 1
            continue
        
        # Extract the subquery (without outer parentheses)
        subquery_sql = modified_sql[start_pos + 1:end_pos - 1].strip()
        
        # Recursively parse the subquery
        parsed = parse_sql_nested(subquery_sql, headerDic, tableDic)
        
        if parsed:
            # Determine location (WHERE, FROM, SELECT)
            location = 'unknown'
            prefix = modified_sql[:start_pos].lower()
            
            # Look at the words immediately before the subquery
            prefix_words = prefix.split()[-5:] if prefix.split() else []
            
            if 'in' in prefix_words or 'where' in prefix_words:
                location = 'where'
            elif 'from' in prefix_words:
                location = 'from'
            elif 'select' in prefix_words:
                location = 'select'
            
            subqueries.append({
                'location': location,
                'query': parsed
            })
        
        # Replace with placeholder
        placeholder = f' SUBQUERY_{len(subqueries)} '
        modified_sql = modified_sql[:start_pos] + placeholder + modified_sql[end_pos:]
        
        # Continue searching from after the placeholder
        pos = start_pos + len(placeholder)
    
    return modified_sql


def parse_single_query(sql: str, headerDic: List[str], tableDic: List[str]) -> Dict:
    """
    Parse a single SQL query (without nested subqueries).
    This is similar to your original parse_sql function.
    """
    sqlForm = {
        'sel': '',
        'agg': [],
        'tab': [],
        'cond': []
    }
    
    # Remove ORDER BY, LIMIT, GROUP BY clauses before parsing
    sql = re.sub(r'\border\s+by\s+[^)]*?(?=where|from|and|$|\))', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\blimit\s+\d+', '', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bgroup\s+by\s+[^)]*?(?=where|from|order|limit|$|\))', '', sql, flags=re.IGNORECASE)
    
    # Split by WHERE
    arr = re.split(r'\bwhere\b', sql, flags=re.IGNORECASE)
    
    # Split SELECT and FROM
    qlead = re.split(r'\bfrom\b', arr[0], flags=re.IGNORECASE)
    
    # Clean up SELECT clause
    qlead[0] = ",".join([i.strip() for i in qlead[0].split(",")])
    
    # Remove DISTINCT keyword if present
    qlead[0] = re.sub(r'\bdistinct\b', '', qlead[0], flags=re.IGNORECASE)
    
    qagg = re.split(r'\s+', qlead[0])
    qagg = list(filter(None, qagg))

    # Normalize tokens (handle parentheses and commas)
    normalized = []
    for token in qagg:
        parts = re.findall(r'[^\(\),]+|[\(\),]', token)
        normalized.extend(part for part in parts if part.strip())
    qagg = normalized

    # Extract aggregation operation
    agg_ops = ['count', 'min', 'max', 'avg', 'sum']
    for token in qagg:
        if token in agg_ops:
            sqlForm['sel'] = token
            break

    # Extract columns from SELECT
    # Look for actual column references (table.column or just column)
    for wd in qagg:
        # Check if it's in headerDic
        if wd in headerDic:
            if wd not in sqlForm['agg']:
                sqlForm['agg'].append(wd)
        # Check if it's an alias.column pattern (like t1.subject_id)
        elif '.' in wd:
            # Check if the column part matches any header
            alias, col = wd.rsplit('.', 1)
            # See if any header ends with this column name
            for header in headerDic:
                if header.endswith('.' + col):
                    if wd not in sqlForm['agg']:
                        sqlForm['agg'].append(wd)
                    break
    
    # Extract tables from FROM
    if len(qlead) > 1:
        from_clause = qlead[1].strip()
        
        # Remove AS aliases (e.g., "SUBQUERY_1 as t1" -> "SUBQUERY_1")
        from_clause = re.sub(r'\bas\s+\w+', '', from_clause, flags=re.IGNORECASE)
        
        qtab = re.split(r'\s+', from_clause)
        qtab = list(filter(None, qtab))
        
        for wd in qtab:
            clean_wd = wd.strip(',').strip()
            # Check if it's a real table or a SUBQUERY placeholder
            if clean_wd in tableDic:
                if clean_wd not in sqlForm['tab']:
                    sqlForm['tab'].append(clean_wd)
            # Also accept SUBQUERY_ placeholders as "tables"
            elif clean_wd.startswith('SUBQUERY_'):
                if clean_wd not in sqlForm['tab']:
                    sqlForm['tab'].append(clean_wd)
    
    # Extract conditions from WHERE
    if len(arr) > 1:
        where_clause = arr[-1]
        
        # Split by AND (but be careful with "IS NOT")
        qtail = re.split(r'\band\b', where_clause, flags=re.IGNORECASE)
        
        for cond in qtail:
            cond = cond.strip()
            
            # Skip BETWEEN clauses for now (they're complex)
            if 'between' in cond.lower():
                continue
            
            # Handle "IS NOT NULL" and "IS NULL" specially
            is_not_match = re.match(r'(\S+)\s+is\s+not\s+null', cond, re.IGNORECASE)
            is_null_match = re.match(r'(\S+)\s+is\s+null', cond, re.IGNORECASE)
            
            if is_not_match:
                col = is_not_match.group(1)
                if col in headerDic or any(col.endswith('.' + h.split('.')[-1]) for h in headerDic) or '.' in col:
                    sqlForm['cond'].append([col, 'is', 'not null'])
                continue
            elif is_null_match:
                col = is_null_match.group(1)
                if col in headerDic or any(col.endswith('.' + h.split('.')[-1]) for h in headerDic) or '.' in col:
                    sqlForm['cond'].append([col, 'is', 'null'])
                continue
            
            # Handle regular conditions with operators
            # Look for operators: =, !=, <, >, <=, >=, LIKE, IN, etc.
            op_pattern = r'\s+(<=|>=|!=|<>|=|<|>|like|in|not\s+in)\s+'
            match = re.search(op_pattern, cond, re.IGNORECASE)
            
            if match:
                op = match.group(1).strip().lower()
                parts = cond.split(match.group(0), 1)
                
                if len(parts) == 2:
                    col_part = parts[0].strip()
                    val_part = parts[1].strip()
                    
                    # Normalize whitespace in column part (for functions like strftime)
                    # Remove spaces after commas and inside parentheses
                    col_part = re.sub(r',\s+', ',', col_part)
                    col_part = re.sub(r'\(\s+', '(', col_part)
                    col_part = re.sub(r'\s+\)', ')', col_part)
                    
                    # Clean up value - but preserve it mostly as-is
                    # Remove only leading/trailing quotes, not internal content
                    val_part = val_part.strip('"').strip("'")
                    
                    # Accept column if it's in headerDic, matches pattern, or is an alias.column or function call
                    if (col_part in headerDic or 
                        any(col_part.endswith('.' + h.split('.')[-1]) for h in headerDic) or
                        '.' in col_part or
                        '(' in col_part):  # Accept function calls
                        sqlForm['cond'].append([col_part, op, val_part])
    
    # Sort conditions for consistent comparison
    sqlForm['cond'] = sorted(sqlForm['cond'])
    
    return sqlForm


def flatten_logical_form(lf: Dict) -> Dict:
    """
    Recursively flatten a logical form by collecting all elements from subqueries.
    Returns a flattened version with all tabs, aggs, conds from nested queries.
    """
    if lf is None:
        return None
    
    flattened = {
        'sel': lf['sel'],
        'agg': list(lf['agg']),  # Make a copy
        'tab': list(lf['tab']),  # Make a copy
        'cond': list(lf['cond'])  # Make a copy
    }
    
    # Recursively add elements from subqueries
    for subquery_info in lf.get('subqueries', []):
        sub_lf = subquery_info['query']
        sub_flattened = flatten_logical_form(sub_lf)
        
        if sub_flattened:
            # Add tables from subquery
            for tab in sub_flattened['tab']:
                if tab not in flattened['tab']:
                    flattened['tab'].append(tab)
            
            # Add aggregated columns from subquery
            for agg in sub_flattened['agg']:
                if agg not in flattened['agg']:
                    flattened['agg'].append(agg)
            
            # Add conditions from subquery
            for cond in sub_flattened['cond']:
                if cond not in flattened['cond']:
                    flattened['cond'].append(cond)
    
    # Remove SUBQUERY_ placeholders from tables
    flattened['tab'] = [t for t in flattened['tab'] if not t.startswith('SUBQUERY_')]
    
    # Remove conditions that reference SUBQUERY_ placeholders
    flattened['cond'] = [c for c in flattened['cond'] if not any('SUBQUERY_' in str(part) for part in c)]
    
    # Sort for consistent comparison
    flattened['tab'] = sorted(flattened['tab'])
    flattened['agg'] = sorted(flattened['agg'])
    flattened['cond'] = sorted(flattened['cond'])
    
    return flattened


def logic_form_accuracy(result_dataset: List[Dict], db_model) -> Dict[str, float]:
    """
    Extended version of your original function with nested query support.
    Flattens nested queries when computing accuracy scores.
    """
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
        gold_sql = post_process_sql(str(line["gold_sql"]))
        pred_sql = post_process_sql(str(line["predicted_sql"]))

        if pred_sql is None:
            pred_sql = "None"
        
        # Handle impossible queries
        if gold_sql == "null":
            if pred_sql == "None":
                outGen.append(None)
                outTtt.append(None)
            else:
                outGen.append(None)
                outTtt.append("null")
            continue
        
        # Handle empty predictions
        if pred_sql.strip() == '' or pred_sql == 'None':
            pred_sql = 'SELECT TEST."NOTHING" FROM TEST WHERE TEST."NOTHING" = "NONE"'
        
        gen = re.split('<stop>', pred_sql)[0]
        
        # Use nested parser
        sqlG = parse_sql_nested(gen, headerDic, tableDic)
        outGen.append(sqlG)
        
        sqlT = parse_sql_nested(gold_sql, headerDic, tableDic)
        outTtt.append(sqlT)

    lf_count = {
        "total": 0,
        "agg_op": 0,
        "agg_col": 0,
        "table": 0,
        "condition_column_operation": 0,
        "condition_value": 0,
    }

    for k in range(len(outGen)):
        gen_lf = outGen[k]
        gold_lf = outTtt[k]

        # Both None → correct
        if gen_lf is None and gold_lf is None:
            lf_count["total"] += 1
            lf_count["agg_op"] += 1
            lf_count["agg_col"] += 1
            lf_count["table"] += 1
            lf_count["condition_column_operation"] += 1
            lf_count["condition_value"] += 1
            continue

        # One None → wrong
        if gen_lf is None or gold_lf is None:
            continue

        # Flatten both logical forms for comparison
        gen_flat = flatten_logical_form(gen_lf)
        gold_flat = flatten_logical_form(gold_lf)

        # Check exact match (flattened)
        if (gen_flat['sel'] == gold_flat['sel'] and
            gen_flat['agg'] == gold_flat['agg'] and
            gen_flat['tab'] == gold_flat['tab'] and
            gen_flat['cond'] == gold_flat['cond']):
            lf_count["total"] += 1
        else:
            # Debug output
            if result_dataset[k]["predicted_sql"] != result_dataset[k]["gold_sql"]:
                print(result_dataset[k]["question"])
                print("-"*10)
                print("PREDS:")
                print(result_dataset[k]["predicted_sql"])                
                print("Flattened:", gen_flat)
                print("-"*10)
                print("TRUTH:")
                print(result_dataset[k]["gold_sql"])
                print("Flattened:", gold_flat)
                print("-"*10)
                print()

        # Component-wise checks using flattened forms
        if gen_flat['sel'] == gold_flat['sel']:
            lf_count["agg_op"] += 1 

        if gen_flat['agg'] == gold_flat['agg']:
            lf_count["agg_col"] += 1

        if gen_flat['tab'] == gold_flat['tab']:
            lf_count["table"] += 1 

        arrG = [wd[:2] for wd in gen_flat['cond']]
        arrT = [wd[:2] for wd in gold_flat['cond']]
        if arrG == arrT:
            lf_count["condition_column_operation"] += 1

        arrG = [wd[:3] for wd in gen_flat['cond']]
        arrT = [wd[:3] for wd in gold_flat['cond']]
        if arrG == arrT:
            lf_count["condition_value"] += 1

    return {cat: (cnt / len(outGen)) if len(outGen) > 0 else 0.0 
            for (cat, cnt) in lf_count.items()}



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
        gold_sql = post_process_sql(str(item["gold_sql"]))
        pred_sql = post_process_sql(str(item["predicted_sql"]))
        
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