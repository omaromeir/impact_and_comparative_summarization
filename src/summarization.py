import numpy as np  
import pandas as pd  
import math
import pydot
from operator import attrgetter
import time
import random
import re
# Parse sql
import sqlparse
from sqlparse.sql import Where, Comparison, Parenthesis, TokenList, Token, IdentifierList, Identifier

'''
The summarization step to process provenence data before generating rules
'''

PREFIX_STR = 'prov_public_'

def getKeyMapping(query, re_str, split_str):
    attrs = re.findall(re_str, query)
    mapping = {}
    for i in range(len(attrs)):
        str1 = attrs[i]
        attr = str1.split(split_str)[1]
        mapping[attr] = str1
    return mapping

def getRenameMapping(query, re_str, split_str):
    attrs = re.findall(re_str, query)
    mapping = {}
    for i in range(len(attrs)):
        str1 = attrs[i]
        str1 = str1.split(split_str)
        mapping[str1[1]] = str1[0]
    return mapping

def getConditions(t, mapping):
    # Get the additional conditions with the selected t
    condition_str = " "
    join_keys = {}
    for i in range(len(t)):
        col_name = t[i][0]
        value = t[i][1]
        if not re.match(r'(count|sum|avg|min|max)$', col_name):
            if col_name in mapping.keys():
                col_name = mapping[col_name]
            condition_str = condition_str + "AND " + col_name + " = "
            if (type(t[i][1]) is not str):
                condition_str = condition_str + "\'" + str(value) + "\' "
            else:
                value = '\\\''.join(value.split("\'"))
                condition_str = condition_str + "\'" + re.escape(value) + "\' "
    return condition_str

def findPattern(x, p):
    if re.match(p, x):
        return x

def getSubstringBetween(source, start_sep, end_sep):
    result=[]
    tmp=source.lower().split(start_sep)
    for par in tmp:
        if end_sep in par:
            result.append(par.split(end_sep)[0])
    return result

# Assume there's always two tables
def getTables(query):
    tables = getSubstringBetween(query, 'from', 'where') # Always will have the join condition in WHERE
    tables = [t.split(' ') for t in tables[0].split(',')]
    for i in range(len(tables)):
        tables[i] = [t for t in tables[i] if len(t) > 0] # Remove empty terms
    table_R = tables[0][0]
    table_T = tables[1][0]
    return table_R, table_T

def getProvenance(Q, query_tokens, conn, Agg, target):
    # Get table names
    prefix = PREFIX_STR
    table_R, table_T = getTables(Q)
    table_R = prefix + table_R + '_'
    table_T = prefix + table_T + '_'
    
    # Get provenance
    print(str(query_tokens))
    df = pd.read_sql_query(str(query_tokens), conn)
    cols_R = [x for x in list(df.columns) if findPattern(x, table_R)]
    cols_T = [x for x in list(df.columns) if findPattern(x, table_T)]
    # Rename columns
    df_R = df[cols_R]
    df_R.columns = [x.replace(table_R, '') for x in list(df_R.columns)]
    df_T = df[cols_T]
    df_T_cols = [x.replace(table_T, '') for x in list(df_T.columns)]
    df_T.columns = df_T_cols
    # Augment the target column to table_T
    df_T[target] = df_R[target]
    
    # Aggregate the target value for T
    options = {
        'count' : df_T.groupby(df_T_cols, as_index=False).count(),
        'sum' : df_T.groupby(df_T_cols, as_index=False).sum(),
        'avg' : df_T.groupby(df_T_cols, as_index=False).mean(),
        'max' : df_T.groupby(df_T_cols, as_index=False).max(),
        'min' : df_T.groupby(df_T_cols, as_index=False).min(),
    }
    df_T = options[Agg.lower()]
    
    return df_R, df_T
