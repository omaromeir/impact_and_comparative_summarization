import numpy as np  
import pandas as pd
import psycopg2
import math
import pydot
from operator import attrgetter
import json
import random
import re

from src.rule import Rule
from src.rule import create_rule

from timeit import default_timer as timer

'''
Helper functions for the pastwatch system
'''

def get_init_rules(size, cats):
    '''
    Get size 1 rules, when no sub-rule has been computed yet
    :param size: rule size
    :param cats: list of categories, where the rule value is non-star, same length as size
    '''
    rule_ls = []    
    for c in cats:            
        # initialize idx for every col
        idx = 0    
        for item in cats[c]:            
            while idx < item.size:                
                # var and count specific to a variable of a col
                var = (item[idx],)
                r = Rule()
            
                # if sub_rule is None:
                r.rule = r.rule + (c,)
                r.var = r.var + var
                r.size = size 
                # we don't know score yet, too expensive to calculate

                # save to dict
                rule_ls.append(r)
            
                # increment index idx
                idx = idx+1
    
    return rule_ls

def make_cats_dict(cats):
    '''
    Make a category dictionary to prevent generating the same rule many times
    :param cats: list of categories in the provenence
    '''
    cats_keys = list(cats.keys())
    cats_len = len(cats.keys())
    cats_dict = {}

    for c in range(cats_len):
        cats_dict[cats_keys[c]] = cats_keys[c+1:cats_len]
    return cats_dict

def get_cats_options(sub_r, cats_dict):
    i = len(sub_r.rule)-1
    return cats_dict[sub_r.rule[i]]

def get_sup_rule_ls(size, sub_r, cats, cats_dict, df):
    '''
    Get super-rules for a given sub-rule
    :param cats: list of categories in the provenence
    '''  
    rule_ls = []   
    cats = get_cats_options(sub_r, cats_dict)

    for c in cats:        
        # check if the col is already in the super rule
        if c not in sub_r.rule:        
            
            ### write a function to replace cats[c]: existing combinations for a given col
            # st = timer()
            sup_elements = get_sup_rule_elements(c, sub_r, df)
            # en = timer()
            # print(en-st)
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

def get_super_rules(size, sub_rules, cats, cats_dict, df):
    '''
    Get super-rules given sub-rules list
    :param cats: list of categories in the provenence
    '''
    rule_ls = []

    for sub_r in sub_rules: 
        rule_ls = rule_ls + get_sup_rule_ls(size, sub_r, cats, cats_dict, df)

    return rule_ls

def find_max_score(C, S):
    '''
    Find the rule with max score
    :param S: the result rule set
    :param C: candidate rules
    '''
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
    '''
    Check if a rule is already in the result rule set
    :param r: a rule
    :param S: the result rule set
    '''
    for s in S:
        if equalRules(r, s): return True
    return False

def equalRules(r1, r2):
    '''
    Check if two rules have the same non-star attributes and values
    :param r1: a rule
    :param r2: a rule
    '''
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

def W_size(R):
    '''
    A weighing function based on the size of a rule
    :param R: a rule
    '''
    if R:
        return R.size

def printRules(rules):
    for r in rules:
        r.printRule()

def sort_by_weight(rules):
    '''
    Sort the rule set by weight in descending order
    :param rules: the rule set
	'''
    return sorted(rules, key = attrgetter('size'), reverse = True)

def filter_by_rule(R, df):
    '''
    Get rows in a table based on the attributes and values of a rule
    :param R: a rule
    :param df: provenance table
    '''
    r_filter = [1] * len(df)
    for col in range(len(R.rule)):
        temp = df[R.rule[col]] == R.var[col]
        r_filter = r_filter & temp
    return df[r_filter]

def get_marginal(S, C_n, W, Agg, target, df):
    '''
    Get the marginal score of each rule
    :param S: the result rule list so far
    :param C_n: candidate rule list
    :param W: weighing function
    :param Agg: the aggregate function
    :param target: the target attribute to summarize over
    :param df: the provenence dataframe
    '''
    S = sort_by_weight(S)
    if not S:
        # for each row
        for row in df.itertuples():
            for R in C_n:
                if_R_covers = True
                # get marginal score
                for i in range(len(R.rule)):
                    if getattr(row, R.rule[i]) != R.var[i]:
                        if_R_covers = False
                if if_R_covers:
                    impact_row = getattr(row, 'per_row_impact')
                    R.count = R.count + 1
                    R.abs_count = R.abs_count + 1
                    R.impact = R.impact + impact_row
                    R.score = R.score + impact_row * W(R)
                    R.abs_score = R.abs_score + R.score
    
    else:  
        # for each row
        for row in df.itertuples():
            highest_s = Rule()
            if_S_covers = True
            for R_s in S:
                # find if covers
                for i in range(len(R_s.rule)):
                    if getattr(row, R_s.rule[i]) != R_s.var[i]:
                        if_S_covers = False
                # set the highest weight rule
                if if_S_covers:
                    highest_s = R_s
                    break
    
            # for each rule in C_n that covers t
            for R in C_n:
                if_R_covers = True
                for i in range(len(R.rule)):
                    if getattr(row, R.rule[i]) != R.var[i]:
                        if_R_covers = False
                if if_R_covers:
                    impact_row = getattr(row, 'per_row_impact')
                    R.abs_count = R.abs_count + 1
                    R.impact = R.impact + impact_row
                    R.abs_score = R.abs_score + impact_row * W(R)
                    if not if_S_covers:
                        R.score = R.score + impact_row * W(R)
                        R.count = R.count + 1
                    # If t is covered by s, compare the weights of R and s 
                    elif W(R) > W(highest_s):
                        R.score = R.score + impact_row * (W(R)-W(highest_s))
                        # Do nothing when W(R) <= W(highest_s)

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

            # map the inputs to the function blocks
            options = {
                'count' : get_count,
                'sum' : get_sum,
                'avg' : get_avg,
                # 'max' : get_max,
                # 'min' : get_min,
            }
            options[Agg.lower()](S, C_n, C, max_w, W, Agg, target, df, H)

            # if C_n is empty
            if not C_n:
                break

            C = C + C_n
            C_o = C_n
            C_n = []
            H, BMR = find_max_score(C, S)       
    
    return BMR

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

def BRS(k, cats, cats_dict, max_w, W, Agg, target, df):
    '''
    Find the best rule set
    :param k: the length of the result rule set
    :param cats: attribute list from the provenance 
    :param cats_dict: attribute disctionary
    :param max_w: maximum weight
    :param W: weighing function
    :param Agg: aggregation function
    :param target: the target attribute to summarize over
    :param df: provenance dataframe
    '''
    S = []
    for i in range(k):
        # cats and query should be obtained from T (database table) instead
        R_m = find_best_marginal_rule(S, cats, cats_dict, max_w, W, Agg, target, df)
        # R_m.printRule()
        # If R_m is empty, fill the rest rules with empty rules
        if R_m.size == 0:
            for j in range(k-i):
                S.append(Rule())
            return S
        else:
            S.append(R_m)
    return S

def makeCats(df, cols):
    '''
    Get attributes and their values from a given provenance table 
    :param df: the provenance table
    :param cols: desired columns in the table
    '''
    cats = {}
    for i in range(len(cols)):
        cats[cols[i]] = []
        cats[cols[i]].append(df[cols[i]].unique())
    
    cats_dict = make_cats_dict(cats)
    
    return df, cats, cats_dict
