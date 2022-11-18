import numpy as np
import pandas as pd
import psycopg2
import networkx as nx
import math
import pydot
from operator import attrgetter
import json
import random
import re

# rule class
from src.rule import create_rule
# summarization methods
from src.summarization import getKeyMapping, getConditions, findPattern
# pastwatch utils
from src.utils import W_size, get_init_rules, make_cats_dict, get_super_rules, find_max_score, printRules, sort_by_weight, filter_by_rule, get_marginal, BRS, makeCats

from timeit import default_timer as timer

######################################################################
# summarization(D, Q, t, k)
# D is the database, Q is the query, t is the randomly selected tuple, k is number of rules
def summarization(conn, Q, t, prefix):
    
    # Get the additional conditions with the selected t
    key_mapping = getKeyMapping(Q, r'[a-zA-Z]+\.[a-zA-Z_]+', '.')
    condition_str = getConditions(t, key_mapping)
    
    # Parse the query to get the WHERE clause
    query_tokens = sqlparse.parse(Q)[0]
    where = [token for token in query_tokens.tokens if isinstance(token, Where)]
    groupby = [token for token in query_tokens.tokens if token.value.lower() == 'group by']
    
    # Check the presence of the WHERE clause
    if not where:
        con_tokens = sqlparse.parse(condition_str.replace('AND', 'WHERE', 1))[0]
        TokenList.insert_before(self=query_tokens,where=next(iter(groupby)), token=con_tokens)
    else: 
        # Tokenize the additional conditions
        con_tokens = sqlparse.parse(condition_str)[0]
        # Insert it to the WHERE clause of the user input query 
        TokenList.insert_after(self=query_tokens,where=next(iter(where)), token=con_tokens, skip_ws=True)
    
    # Insert provenance keyword 
    p_token = sqlparse.parse('provenance ')[0]
    select = next(token for token in query_tokens.tokens if token.value.lower() == 'select')
    TokenList.insert_after(self=query_tokens,where=select, token=p_token, skip_ws=True)
    
    # Get provenance
    # print(str(query_tokens))
    df = pd.read_sql_query(str(query_tokens), conn)
    cols = [x for x in list(df.columns) if findPattern(x, prefix)]
    # Rename columns
    df = df[cols]
    df.columns = [x.replace(prefix, '') for x in list(df.columns)]
    
    return df

        
######################################################################
# methods
def get_min_or_max(S, C_n, W, Agg, target, df, values):
    first = values[0]
    second = values[1]
    diff = abs(first - second)
    row_count = df.loc[df[target] == first].shape[0]
    
    # Set impact per row
    df_m = df.loc[df[target] == first].copy()
    df_m['per_row_impact'] = diff / row_count

    # Get marginal score and impact of C_n
    get_marginal(S, C_n, W, Agg, target, df_m)

def get_max(S, C_n, C, max_w, W, Agg, target, df, H):
    targets = list(df[target].drop_duplicates())
    if len(targets) < 2:
        values = [targets[0], 0]
    else: 
        values = sorted(targets, reverse=True)[0:2]
    get_min_or_max(S, C_n, W, Agg, target, df, values)

def get_min(S, C_n, C, max_w, W, Agg, target, df, H):
    #values = list(df[target].drop_duplicates().nsmallest(2))
    targets = list(df[target].drop_duplicates())
    if len(targets) < 2:
        values = [targets[0], 0]
    else: 
        values = sorted(targets)[0:2]
    get_min_or_max(S, C_n, W, Agg, target, df, values)

def get_avg(S, C_n, C, max_w, W, Agg, target, df, H):
    # compute total count and total avg
    if df.shape[0] == 1:
        total_count = 2
    else:
        total_count = df.shape[0]
    total_avg = df[target].mean(axis = 0, skipna = True)
    
    for R in C_n:            
        M = math.inf
        # s is R's sub-rule           
        for s in R.sub_rule:                
            # if R's sub-rule is in C
            if s in C:                    
                # use the sub-rule to calculate the estimated score
                temp = filter_by_rule(s, df)
                temp = temp[target].mean(axis = 0, skipna = True)
                score = s.score + temp * (max_w - W(s))

                M = min(M, score)
                    
        if M < H:
            C_n.remove(R)

    df['per_row_impact'] = df[target].apply(lambda x: abs(total_avg - ((total_avg * total_count)-x)/(total_count-1)))

    # Get marginal score and impact of C_n
    get_marginal(S, C_n, W, Agg, target, df)

def get_sum(S, C_n, C, max_w, W, Agg, target, df, H):
    for R in C_n:            
        M = math.inf
        # s is R's sub-rule           
        for s in R.sub_rule:                
            # if R's sub-rule is in C
            if s in C:                    
                # use the sub-rule to calculate the estimated score
                temp = filter_by_rule(s, df)
                temp = temp[target].sum(axis = 0, skipna = True)
                score = s.score + temp * (max_w - W(s))

                M = min(M, score)
                    
        if M < H:
            C_n.remove(R)

    df['per_row_impact'] = df[target].apply(lambda x: abs(x))

    # Get marginal score and impact of C_n
    get_marginal(S, C_n, W, Agg, target, df)

def get_count(S, C_n, C, max_w, W, Agg, target, df, H):
    for R in C_n:           
        M = math.inf
        # s is R's sub-rule           
        for s in R.sub_rule:                
            # if R's sub-rule is in C
            if s in C:                    
                # use the sub-rule to calculate the estimated score
                temp = filter_by_rule(s, df)
                temp = len(temp)
                score = s.score + temp * (max_w - W(s))

                M = min(M, score)
                    
        if M < H:
            C_n.remove(R)

    df['per_row_impact'] = 1

    # Get marginal score and impact of C_n
    get_marginal(S, C_n, W, Agg, target, df)

# cats and query should be obtained from T (database table) instead
def find_best_marginal_rule(S, cats, cats_dict, max_w, W, Agg, target , df):
    
    H = 0
    BMR = create_rule()
    size = 0
    C = C_o = C_n = []
    
    # for each column
    for c in cats:        
        # increment size
        size = size + 1
        # Drop out if max_w is reached
        if size <= max_w:
            # generate corresponding size rules
            if not C_o:
                C_n = get_init_rules(size, cats)
            else: 
                # C_o is the sub-rules
                C_n = get_super_rules(size, C_o, cats, cats_dict, df)
            
            # if C_n is empty
            if not C_n:
                break
        
            # map the inputs to the function blocks
            options = {
                'count' : get_count,
                'sum' : get_sum,
                'avg' : get_avg,
                'max' : get_max,
                'min' : get_min,
            }
            options[Agg.lower()](S, C_n, C, max_w, W, Agg, target, df, H)

            C = C + C_n
            C_o = C_n
            C_n = []
            H, BMR = find_max_score(C, S)
    
    return BMR

def pastWatch(k, Agg, target, query, conn, cols, max_w, sz, W = W_size):
    # get data frame
    # df = summarization(conn, query, t, prefix) # glei_asp:'prov_public_glei_', tpch_aspj:'prov_public_', tpch_asp:'prov_public_lineitem_'
    # sz is sample size
    df = pd.read_sql_query(query, conn)
    df = df.loc[:,~df.columns.duplicated()]
    df = df[pd.notnull(df[target])]
    if sz > 0:
        if sz < len(df):
            df = df.sample(n = sz, random_state=1)

    # print(df['original_language'])
    # print(df['production_country'])
    
    # get categorical columns and their variables
    df, cats, cats_dict = makeCats(df, cols)
    
    # Set max_w
    if max_w > len(cats):
        max_w = len(cats)
        print('Warning: max_w is greater than the number of categorical columns in the data. Use len(cats) instead, where max_w is set to:', max_w)
    
    # get best rule set
    S = BRS(k, cats, cats_dict, max_w, W, Agg, target, df)
    
    return df.shape[0], S