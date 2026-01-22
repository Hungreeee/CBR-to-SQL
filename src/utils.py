import re
import sqlite3
import random
from collections import defaultdict


def tokenize(text: str):
    return re.findall(r'\b\w+\b', text.lower())


def remove_sql_wrapper(sql_query: str | None) -> str | None:
    if sql_query is None: 
        return None
    pattern = r"```sql\s*(.*?)\s*```"
    return re.sub(pattern, r"\1", sql_query, flags=re.DOTALL | re.IGNORECASE)


def drop_cases(cases, top_k=5, p_top=1.0):
    def drop_prob(rank):
        if rank > top_k:
            return 0.0
        return p_top * (top_k - rank) / (top_k - 1)
    
    filtered = [
        case for rank, case in enumerate(cases, 1)
        if random.random() > drop_prob(rank)
    ]
    return filtered


class query(object):
    
    def __init__(self, db_file):
        self.db_meta, self.db_tabs, self.db_head = self._load_db(db_file)
        self.agg_op = ['', 'count', 'max', 'min', 'avg']
        self.cond_op = ['=', '>', '<', '>=', '<=']
    
    def __call__(self, sql_):
        '''
        select $$$ ### from *** where ===
        '''
        '''###'''
        mm_agg_col = []
        for itm in sql_['agg_col']:
            tt = self.db_tabs[itm[0]]
            hh = self.db_head[tt][itm[1]]
            mm_agg_col.append('.'.join([tt, hh]))
        mm_agg_col = ','.join(mm_agg_col)
        '''$$$'''
        if sql_['sel'] == 0:
            mm_agg = '{}'.format(mm_agg_col)
        elif sql_['sel'] == 1:
            mm_agg = 'COUNT ( DISTINCT {} )'.format(mm_agg_col)
        elif sql_['sel'] == 2:
            mm_agg = 'MAX ( {} )'.format(mm_agg_col)
        elif sql_['sel'] == 3:
            mm_agg = 'MIN ( {} )'.format(mm_agg_col)
        elif sql_['sel'] == 4:
            mm_agg = 'AVG ( {} )'.format(mm_agg_col)
        '''***'''
        tbtb = [self.db_tabs[k] for k in sql_['table']]
        mm_tab = [tbtb[0]]
        for k in range(1, len(tbtb)):
            mm_tab.append('INNER JOIN')
            mm_tab.append(tbtb[k])
            mm_tab.append('on')
            mm_tab.append('{}.{} = {}.{}'.format(tbtb[0], 'HADM_ID', tbtb[k], 'HADM_ID'))
        '''==='''
        mm_cond = []
        for itm in sql_['cond']:
            tt = self.db_tabs[itm[0]]
            cc = self.db_head[tt][itm[1]]
            oo = self.cond_op[itm[2]]
            ff = itm[3]
            cond1 = '{}.{} {} {}'.format(tt, cc, oo, '"'+str(ff)+'"')
            mm_cond.append(cond1)
        mm_cond = ' AND '.join(mm_cond)
        bb_query = 'SELECT {} FROM {} WHERE {}'.format(mm_agg, ' '.join(mm_tab), mm_cond)
                
        return bb_query
    
    def _load_db(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.cur = self.conn.cursor()
        self.cur.execute("select * from sqlite_master where type='table';")
        results = self.cur.fetchall()
        db_meta = {}
        db_tabs = []
        db_head = {}

        for tb in results:
            table_name = tb[2]
            create_stmt = tb[4]  # SQL statement is at index 4
            
            db_meta[table_name] = {}
            db_tabs.append(table_name)
            db_head[table_name] = {}
            
            # Split and filter the CREATE TABLE statement
            arr = re.split('\n', create_stmt)
            arr = [line.strip() for line in arr if line.strip() and not line.strip().startswith('CREATE') and not line.strip() == ')']
            
            dbaa = []
            for itm in arr:
                # Remove trailing commas and parentheses
                itm = itm.rstrip(',').rstrip(')')
                ttl = re.split('\s+', itm.strip())
                ttl = list(filter(None, ttl))
                
                if len(ttl) >= 2:  # Ensure we have at least column name and type
                    col_name = ttl[0]
                    col_type = ttl[1]
                    db_meta[table_name][col_name] = col_type
                    dbaa.append(col_name)
            
            db_head[table_name] = dbaa

        return (db_meta, db_tabs, db_head)
    
    def execute_sql(self, sql_):
        return self.cur.execute(sql_)


def get_value_pool_(db_file, model, samp_cond):
    (db_meta, db_tabs, db_head) = model._load_db(db_file)
    pool_ = []
    for itm in samp_cond:
        mytb = db_tabs[itm[0]]
        myhd = db_head[mytb][itm[1]]
        mysql = 'select {} from {}'.format(myhd, mytb)
        myres = model.execute_sql(mysql).fetchall()
        myres = list({k[0]: {} for k in myres})
        pool_.append(myres)
        
    return pool_


def is_content_filter_error(error: Exception) -> bool:
    """Check if error is from Azure content filter"""
    error_msg = str(error)
    return (
        "content_filter" in error_msg 
        or "ResponsibleAIPolicyViolation" in error_msg
        or "BadRequestError" in str(type(error))
    )


def stratified_sample(data, p=0.3):
    strata = defaultdict(list)
    for item in data:
        key = item.get('importance', 'impossible')
        strata[key].append(item)
    
    sampled = []
    for items in strata.values():
        sampled.extend(random.sample(items, max(1, int(len(items) * p))))
    
    return sampled