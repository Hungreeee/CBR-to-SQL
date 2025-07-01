from typing import List, Dict
from langchain_community.utilities.sql_database import SQLDatabase
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


def normalize_select_from_clauses(sql: str) -> str:
    sql = sql.lower()
    sql = re.sub(r'\b(count|avg|min|max|sum)\s+\(', r'\1(', sql)
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'select\s+', 'select ', sql)
    sql = re.sub(r'from\s+', 'from ', sql)
    sql = re.sub(r"(?<!\w)'((?:[^']|(?<=\\)')*?)'(?!\w)", r'"\1"', sql)

    return sql.strip()


def logic_form_accuracy(lookup: Dict, result_dataset: List[Dict]):
    headerDic = []
    for tb in lookup:
        for hd in lookup[tb]:
            headerDic.append('.'.join([tb, hd]).lower())

    sql_rec = []

    for line in result_dataset:
        pred = normalize_select_from_clauses(re.split('<stop>', line["sql_query"])[0])
        ttt = normalize_select_from_clauses(line["golden_sql_query"].lower())

        predArr = re.split("where", pred)
        predAgg = re.split("\s", predArr[0])
        predAgg = list(filter(None, predAgg))
        predAgg2 = []
        for k in range(len(predAgg)-1):
            if predAgg[k] in headerDic and predAgg[k+1] in headerDic:
                predAgg2.append(predAgg[k] + ',')
            else:
                predAgg2.append(predAgg[k])
        predAgg2.append(predAgg[-1])
        predAgg = ' '.join(predAgg2)
        
        predCon = re.split("and", predArr[1])
        predConNew = []
        k = 0
        while k < len(predCon):
            if "=" in predCon[k] or "<" in predCon[k] or ">" in predCon[k] or "is" in predCon[k] or "like" in predCon[k] or "in" in predCon[k]:
                predConNew.append(predCon[k])
            else:
                predConNew[-1] += " and " + predCon[k]
                k += 1
            k += 1
        for k in range(len(predConNew)):
            if "=" in predConNew[k]:
                conOp = "="
            if ">" in predConNew[k]:
                conOp = ">"
            if "<" in predConNew[k]:
                conOp = "<"
            if "<=" in predConNew[k]:
                conOp = "<="
            if ">=" in predConNew[k]:
                conOp = ">="
            conVal = re.split("=|<|>", predConNew[k])
            conVal = list(filter(None, conVal))
            conCol = conVal[0]
            conColArr = re.split('\.|\s', conCol)
            conColArr = list(filter(None, conColArr))
            try:
                pool_ = lookup[conColArr[0].upper()][conColArr[1].upper()]
            except:
                sql_rec.append(["Error", ttt])
                continue
            conVal = re.split('"|\s', conVal[-1])
            conVal = list(filter(None, conVal))
            conVal = ' '.join(conVal)
            predConNew[k] = conCol + conOp + ' "' + conVal + '"'

        pred = predAgg + ' where ' + ' and '.join(predConNew)
        pred = re.split("\s", pred)
        pred = list(filter(None, pred))
        pred = " ".join(pred)
        
        sql_rec.append([pred, ttt])

    correct = 0
    for item in sql_rec:
        arr_pred = re.split(',|\s', item[0].lower())
        arr_pred = ' '.join(list(filter(None, arr_pred)))

        arr_gold = re.split(',|\s', item[1].lower())
        arr_gold = ' '.join(list(filter(None, arr_gold)))
        
        if arr_pred == arr_gold:
            correct += 1
        else:
            print(f"PRED: {arr_pred}\nGOLD: {arr_gold}\n")
            print()

    accuracy = correct / len(sql_rec)
    return accuracy
