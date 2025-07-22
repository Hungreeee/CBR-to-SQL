from typing import List, Dict
from langchain_community.utilities.sql_database import SQLDatabase
from src.utils import *
import re

def execution_accuracy(sql_db: SQLDatabase, results_dataset: List[Dict]):
    count = 0

    for item in results_dataset:
        ttt = item["golden_sql_query"]
        outPred = item["sql_response"]

        try:
            outTtt = sql_db.run(ttt)
        except:
            continue
        if outPred == outTtt:
            count += 1
        else:
            print(f"PRED: {item['sql_query']}\nGOLD: {item['golden_sql_query']}\n")
            print()

    accuracy = count / len(results_dataset)
    return accuracy


def parse_sql(sql, headerDic, tableDic):
    sqlForm = {}
    
    arr = re.split('where', sql.lower())
    qlead = re.split('from', arr[0])
    qagg = re.split('\s', qlead[0])
    qagg = list(filter(None, qagg))
    if qagg[1] == 'count' or qagg[1] == 'min' or qagg[1] == 'max' or qagg[1] == 'avg':
        sqlForm['sel'] = qagg[1]
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


def logic_form_accuracy(result_dataset: List[Dict]):
    db_file = './data/TREQS/mimic_db/mimic.db'
    model = query(db_file)
    (_, _, db_head) = model._load_db(db_file)

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
        gen = re.split('<stop>', line['sql_query'])[0]
        sqlG = parse_sql(gen, headerDic, tableDic)
        outGen.append(sqlG)
        
        ttt = line['golden_sql_query']
        sqlT = parse_sql(ttt, headerDic, tableDic)
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
        if outGen[k] == outTtt[k]:
            lf_count["total"] += 1 

        if outGen[k]['sel'] == outTtt[k]['sel']:
            lf_count["agg_op"] += 1 

        if outGen[k]['agg'] == outTtt[k]['agg']:
            lf_count["agg_col"] += 1

        if outGen[k]['tab'] == outTtt[k]['tab']:
            lf_count["table"] += 1 

        arrG = [wd[:2] for wd in outGen[k]['cond']]
        arrT = [wd[:2] for wd in outTtt[k]['cond']]
        if arrG == arrT:
            lf_count["condition_column_operation"] += 1

        arrG = [wd[:3] for wd in outGen[k]['cond']]
        arrT = [wd[:3] for wd in outTtt[k]['cond']]
        if arrG == arrT:
            lf_count["condition_value"] += 1 

    return {cat: (cnt/len(outGen)) for (cat, cnt) in lf_count.items()}

