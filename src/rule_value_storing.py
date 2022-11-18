# basics
import os
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
#from apyori import apriori
import psycopg2
from kmodes.kmodes import KModes
import networkx as nx
from anytree import Node, RenderTree
import math
import pydot
import time
import csv
import json

def convert(o):
    print(o)
    if isinstance(o, np.int64): 
        return int(o)  
    raise TypeError

# make_rules_JSON
def make_rules_JSON(S, attr_dict):
    rules = {}
    count = 0
    for R in S:
        # increment count and make the key
        count = count + 1
        R_key = 'rule' + str(count)
    
        R_temp = {}
        num_attr = len(attr_dict.keys())
        r = ['*'] * num_attr
    
        # append the cols and values of this rule
        for i in range(len(R.var)):
            index = attr_dict[R.rule[i]]
            r[index] = R.var[i]
        
        R_temp['rule'] = r
    
        # put this rule to the dict
        rules[R_key] = [R_temp]
    
        # append the size and score
        rules[R_key].append({
            'size': R.size
        })

        rules[R_key].append({
            'score': R.score
        })

        rules[R_key].append({
            'impact': R.impact
        })

        rules[R_key].append({
            'abs_score': R.abs_score
        })

        rules[R_key].append({
            'm_coverage': R.count
        })

        rules[R_key].append({
            'abs_coverage': R.abs_count
        })
    
    # Return the rules in JSON
    jstr = json.dumps(rules, indent=5, default=convert)
    return jstr

def getAbsCoverage(S):
    abs_coverage = 0
    for R in S:
        abs_coverage += R.abs_count
    return abs_coverage

def getAbsImpact(S):
    abs_impact = 0
    for R in S:
        abs_impact += R.impact
    return abs_impact

def getTotalMarginalCoverage(S):
    m_coverage = 0
    for R in S:
        m_coverage += R.count
    return m_coverage