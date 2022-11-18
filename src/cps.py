import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import psycopg2
from kmodes.kmodes import KModes
import networkx as nx
from anytree import Node, RenderTree
import math
import pydot
from operator import attrgetter
import time
import json

import random
import re

# Parse sql
import sqlparse
from sqlparse.sql import Where, Comparison, Parenthesis, TokenList, Token, IdentifierList, Identifier

######################################################################
# create a rule class

class Rule:
    
    def __init__(self, rule, var, size, count, abs_count, impact, score, abs_score, sub_rule):
        self.rule = rule
        self.var = var
        self.size = size
        self.count = count
        self.abs_count = abs_count
        self.impact = impact
        self.score = score
        self.abs_score = abs_score
        self.sub_rule = sub_rule
    
    def __init__(self):
        self.rule = ()      # tuple of columns                                  e.g., ('gender',)
        self.var = ()       # tuple of the value of corresponding columns       e.g., ('m', )
        self.size = 0       # int
        self.count = 0      # int
        self.abs_count = 0  # int
        self.impact = 0     # int
        self.score = 0      # int
        self.abs_score = 0  # int
        self.sub_rule = []  # list of rules
    
    def printRule(self):
        #print('rule: '+' '.join(self.rule) + ' size:' + str(self.size) +' score:' + str(self.score))
        msg = 'rule: '+' '.join(self.rule) + ' var:'
        for v in range(len(self.var)):        
            if isinstance(self.var[v], str):
                msg = msg + ' ' + self.var[v]
            else: msg = msg + ' ' + str(self.var[v])
        msg = msg + ' size:' + str(self.size) +' score:' + str(self.score) +' impact:' + str(self.impact) +' abs_score:' + str(self.abs_score) + ' marginal_coverage:' + str(self.count) + ' abs_coverage:' + str(self.abs_count)
        print(msg)
        
######################################################################
# summarization methods
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

def get_provenance_query(Q, t):
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
    
    return query_tokens

def get_provenance(conn, query_tokens, prefix):
    print(str(query_tokens))
    df = pd.read_sql_query(str(query_tokens), conn)
    cols = [x for x in list(df.columns) if findPattern(x, prefix)]
    # Rename columns
    df = df[cols]
    df.columns = [x.replace(prefix, '') for x in list(df.columns)]
    
    return df

# summarization(D, Q, t, k)
# D is the database, Q is the query, t is the randomly selected tuple, k is number of rules
def summarization(D, Q, t1, t2, prefix):
    # Connect and load data
    conn = psycopg2.connect(dbname=D, user="perm", host="localhost", port="5432")
    conn.set_client_encoding('UTF8') # Add this line for the lab machine
    df = pd.read_sql_query(Q, conn)
    #df = df[pd.notnull(df[target])]
    print(df.columns)
    
    #t1 = list(zip(df.iloc[t1].index, df.iloc[t1])) # Turn it to a list of tuples
    print(t1)
    #t2 = list(zip(df.iloc[t2].index, df.iloc[t2])) # Turn it to a list of tuples
    print(t2)
    
    # Select a random tuple
    query_tokens1 = get_provenance_query(Q, t1)
    query_tokens2 = get_provenance_query(Q, t2)
    
    # Get provenance
    df1 = get_provenance(conn, query_tokens1, prefix)
    df2 = get_provenance(conn, query_tokens2, prefix)
    
    return df1, df2, str(query_tokens1), str(query_tokens2)

        
######################################################################
# methods

# get size 1 rules, when no sub-rule has been computed yet
def get_init_rules(size, cats):    
    rule_ls = []    
    for c in cats:            
        ### initialize idx for every col
        idx = 0    
        for item in cats[c]:            
            while idx < len(item):                
                ### var and count specific to a variable of a col
                var = (item[idx],)
                r = Rule()
            
                ## if sub_rule is None:
                r.rule = r.rule + (c,)
                r.var = r.var + var
                r.size = size 
                # we don't know score yet, too expensive to calculate

                ### save to dict
                rule_ls.append(r)
            
                # increment index idx
                idx = idx+1
    
    return rule_ls

# to prevent generating the same rule many times
def make_cats_dict(cats):
    cats_keys = list(cats.keys())
    cats_len = len(cats.keys())
    cats_dict = {}

    for c in range(cats_len):
        cats_dict[cats_keys[c]] = cats_keys[c+1:cats_len]
    return cats_dict

def get_cats_options(sub_r, cats_dict):
    i = len(sub_r.rule)-1
    return cats_dict[sub_r.rule[i]]

# helper function to get super-rules for a given sub-rule
def get_sup_rule_ls(size, sub_r, cats, cats_dict, df1, df2):    
    rule_ls = []   
    cats = get_cats_options(sub_r, cats_dict)
    
    for c in cats:        
        # check if the col is already in the super rule
        if c not in sub_r.rule:        
            
            ### write a function to replace cats[c]: existing combinations for a given col
            sup_elements1 = get_sup_rule_elements(c, sub_r, df1)
            sup_elements2 = get_sup_rule_elements(c, sub_r, df2)
            sup_elements = list(set(sup_elements1) & set(sup_elements2))
            
            ## possible variables for that category
            for item in sup_elements:
            
                ### var and count specific to a variable of a col
                var = (item,)
                
                ### create the sub-rule
                r = Rule()
                
                ## if sub_rule is None:
                r.rule = sub_r.rule + (c,)
                r.var = sub_r.var + var
                r.size = size 
                r.sub_rule.append(sub_r)

                ### save to rule list
                rule_ls.append(r)
    
    return rule_ls

def get_sup_rule_elements(c, sub_r, df):
    super_r = sub_r.rule + (c,)
    df_super_r = df[list(super_r)].drop_duplicates()
    
    temp = df_super_r
    for i in range(len(sub_r.rule)):
        temp = temp.loc[temp[sub_r.rule[i]] == sub_r.var[i]]
    
    return list(temp[c])

# get super-rules given sub-rules list
def get_super_rules(size, sub_rules, cats, cats_dict, df1, df2):
    
    rule_ls = []
    
    for sub_r in sub_rules:                
        rule_ls = rule_ls + get_sup_rule_ls(size, sub_r, cats, cats_dict, df1, df2)
    
    return rule_ls

# find the rule with max score
def find_max_score(C, S):
    H = 0
    BMR = Rule()
    for r in C:
        # de-duplicate
        if not inList(r, S):         
            # find max
            if r.score > H:
                H = r.score
                BMR = r
                #r.printRule()
    return H, BMR

def inList(r, S):
    for s in S:
        if equalRules(r, s): return True
    return False

def equalRules(r1, r2):
        if len(r1.rule) != len(r2.rule): return False
            
        for i in range(len(r1.rule)):
            #if r1.rule[i] != r2.rule[i] or r1.var[i] != r2.var[i]:
                #return False
            if r1.rule[i] in r2.rule:
                idx = r2.rule.index(r1.rule[i])
                if r1.var[i] != r2.var[idx]:
                    return False
            else: return False
            
        return True

# weighing function, returns the weight of a rule
def W_size(R):
    if R:
        return R.size

def printRules(rules):
    for r in rules:
        r.printRule()

def sort_by_weight(rules):
    return sorted(rules, key = attrgetter('size'), reverse = True)

def filter_by_rule(R, df):
    r_filter = [1] * len(df)
    for col in range(len(R.rule)):
        temp = df[R.rule[col]] == R.var[col]
        r_filter = r_filter & temp
    return df[r_filter]

def get_marginal(S, C_n, W, Agg, target, df1, df2):
    S = sort_by_weight(S)
    if not S:
        # for each row
        for row1 in df1.itertuples():
            for row2 in df2.itertuples():
                for R in C_n:
                    if_covers = True
                    for i in range(len(R.rule)):
                        row1_var = getattr(row1, R.rule[i])
                        row2_var = getattr(row2, R.rule[i])
                        if row1_var != R.var[i] or row2_var != R.var[i]:
                            if_covers = False
                    if if_covers:
                        #R.score = R.score + getattr(row, target) * (W(R) - min(W(R), W(highest_s)))
                        impact_row = abs(getattr(row1, 'per_row_impact')) + abs(getattr(row2, 'per_row_impact'))
                        R.impact = R.impact + impact_row
                        R.score = R.score + impact_row * W(R)
                        R.count = R.count + 1
                        R.abs_count = R.abs_count + 1
    
    else:
        # for each row
        for row1 in df1.itertuples():
            for row2 in df2.itertuples():
                if_S_covers = True
                if not S: 
                    if_S_covers = False
                else:
                    for R_s in S:
                        # find if covers
                        for i in range(len(R_s.rule)):
                            row1_var = getattr(row1, R_s.rule[i])
                            row2_var = getattr(row2, R_s.rule[i])
                            if row1_var != R_s.var[i] or row2_var != R_s.var[i]:
                                if_S_covers = False
    
                # for each rule in C_n that covers t
                for R in C_n:
                    if_covers = True
                    for i in range(len(R.rule)):
                        row1_var = getattr(row1, R.rule[i])
                        row2_var = getattr(row2, R.rule[i])
                        if row1_var != R.var[i] or row2_var != R.var[i]:
                            if_covers = False
                    if if_covers:
                        impact_row = abs(getattr(row1, 'per_row_impact')) + abs(getattr(row2, 'per_row_impact'))
                        R.impact = R.impact + impact_row
                        R.abs_count = R.abs_count + 1
                        if not if_S_covers:
                            R.score = R.score + impact_row * W(R)
                            R.count = R.count + 1

def get_min_or_max(target, df, values):
    first = values[0]
    second = values[1]
    diff = abs(first - second)
    row_count = df.loc[df[target] == first].shape[0]
    
    # Set impact per row
    df_m = df.loc[df[target] == first].copy()
    df_m['per_row_impact'] = diff / row_count
    
    return df_m

def get_min_values(target, df):
    targets = list(df[target].drop_duplicates())
    if len(targets) < 2:
        values = [targets[0], 0]
    else: 
        values = sorted(targets)[0:2]
    return values

def get_max_values(target, df):
    targets = list(df[target].drop_duplicates())
    if len(targets) < 2:
        values = [targets[0], 0]
    else: 
        values = sorted(targets, reverse=True)[0:2]
    return values

def get_min(target, df1, df2):
    #values = list(df[target].drop_duplicates().nsmallest(2))
    values1 = get_min_values(target, df1)
    values2 = get_min_values(target, df2)
    return get_min_or_max(target, df1, values1), get_min_or_max(target, df2, values2)

def get_max(target, df1, df2):
    values1 = get_max_values(target, df1)
    values2 = get_max_values(target, df2)
    return get_min_or_max(target, df1, values1), get_min_or_max(target, df2, values2)

def get_avg_per_row_impact(target, df):
    # compute total count and total avg
    if df.shape[0] == 1:
        total_count = 2
    else:
        total_count = df.shape[0]
    total_avg = df[target].mean(axis = 0, skipna = True)
    df['per_row_impact'] = df[target].apply(lambda x: abs(total_avg - ((total_avg * total_count)-x)/(total_count-1)))
    return df

def get_avg(target, df1, df2):
    # compute total count and total avg
    df1 = get_avg_per_row_impact(target, df1)
    df2 = get_avg_per_row_impact(target, df2)
    
    return df1, df2

def get_sum(target, df1, df2):
    df1['per_row_impact'] = df1[target].apply(lambda x: abs(x))
    df2['per_row_impact'] = df2[target].apply(lambda x: abs(x))

    return df1, df2

def get_count(target, df1, df2):
    df1['per_row_impact'] = 1
    df2['per_row_impact'] = 1

    return df1, df2

# cats and query should be obtained from T (database table) instead
def find_best_marginal_rule(S, cats, cats_dict, max_w, W, Agg, target, df1, df2):
    
    H = 0
    size = 0
    C = C_o = C_n = []
    
    # for each column
    for c in cats:        
        # increment size
        size = size + 1        
        # generate corresponding size rules
        if not C_o:
            C_n = get_init_rules(size, cats)
        else: 
            # C_o is the sub-rules
            C_n = get_super_rules(size, C_o, cats, cats_dict, df1, df2)
        
        for R in C_n:            
            Ub = math.inf
            # s is R's sub-rule           
            for s in R.sub_rule:                
                # if R's sub-rule is in C
                if s in C:                    
                    # use the sub-rule to calculate the estimated score
                    #df_temp = filter_by_rule(s, df)
                    score = s.score + s.impact * (max_w - W(s))

                    Ub = min(Ub, score)
                    #print('rule: '+' '.join(R.rule) +' M: '+ str(M) + ' score:' + str(score))
                    
            if Ub < H:
                #R.printRule()
                C_n.remove(R)
            
        # if C_n is empty
        if not C_n:
            break
        
        # Get marginal score and impact of C_n
        get_marginal(S, C_n, W, Agg, target, df1, df2)

        C = C + C_n
        C_o = C_n
        C_n = []
        H, BMR = find_max_score(C, S)
    
    return BMR

# cats and query should be obtained from T (database table) instead
def BRS(k, cats, cats_dict, max_w, W, Agg, target, df1, df2):
    S = []
    for i in range(k):
        # cats and query should be obtained from T (database table) instead
        R_m = find_best_marginal_rule(S, cats, cats_dict, max_w, W, Agg, target, df1, df2)
        S.append(R_m)
    return S

# make categories
def makeCats(df1, df2, target, cols):
    # abstract categories
    cats = {}
    for i in range(len(cols)):
        cats[cols[i]] = []
        # Take the intersection of two tables
        intersect = list(set(df1[cols[i]].unique()) & set(df2[cols[i]].unique()))
        cats[cols[i]].append(intersect)
    
    cats_dict = make_cats_dict(cats)
    #print(cats)
    
    return df1, df2, cats, cats_dict

def pastWatch(k, Agg, target, query1, query2, cols, conn, sz, W = W_size):
    
    # # get data frame
    # df1, df2, Q1, Q2 = summarization('testdb', query, t1, t2, 'prov_public_movie_')
    
    # # get categorical columns and their variables
    # df1, df2, cats, cats_dict = makeCats(df1, df2, target, cols)

    df1 = pd.read_sql_query(query1, conn)
    df1 = df1.loc[:,~df1.columns.duplicated()]
    df1 = df1[pd.notnull(df1[target])]
    if sz > 0:
        if sz < len(df1):
            df1 = df1.sample(n = sz, random_state=1)

    df2 = pd.read_sql_query(query2, conn)
    df2 = df2.loc[:,~df2.columns.duplicated()]
    df2 = df2[pd.notnull(df2[target])]
    if sz > 0:
        if sz < len(df2):
            df2 = df2.sample(n = sz, random_state=1)

    df1, df2, cats, cats_dict = makeCats(df1, df2, target, cols)

    # map the inputs to the function blocks
    options = {
        'count' : get_count,
        'sum' : get_sum,
        'avg' : get_avg,
        'max' : get_max,
        'min' : get_min,
    }
    df1, df2 = options[Agg.lower()](target, df1, df2)
    
    # get best rule set
    S = BRS(k, cats, cats_dict, len(cats), W, Agg, target, df1, df2)
    
    return df1.shape[0], df2.shape[0], S
    