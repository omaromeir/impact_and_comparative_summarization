import numpy as np 
'''
Create a rule class
'''

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

    def returnDictRule(self):
        rule = {}
        # rule['number'] = n
        rule['attributes'] = self.rule
        rule['values'] = []
        for v in range(len(self.var)):        
            if isinstance(self.var[v], str):
                rule['values'].append(self.var[v])
            else: rule['values'].append(str(self.var[v]))
        # rule['size'] = str(self.size)
        rule['score'] = str(self.score)
        # rule['impact'] = str(self.impact)
        return rule

    def returnStrRule(self):
        msg = 'rule: '+' '.join(self.rule) + ' var:'
        for v in range(len(self.var)):        
            if isinstance(self.var[v], str):
                msg = msg + ' ' + self.var[v]
            else: msg = msg + ' ' + str(self.var[v])
        msg = msg + ' size:' + str(self.size) +' score:' + str(self.score)
        return msg

    def returnJSONRule(self):
        msg = '{rule: ['
        for a in range(len(self.rule)):
            if isinstance(self.rule[a], str):
                msg = msg + '{' + self.rule[a] + ': '
            if isinstance(self.var[a], str):
                msg = msg + self.var[a] + '} '
            else: msg = msg +  str(self.var[a]) + '}'
            if a < len(self.rule) - 1:
                msg = msg + ','
        msg = msg + '], {size: ' + str(self.size) + '}, {score: ' + str(self.score) + '}}'
        return msg

    def returnTuplesRule(self, n, g):
        rules = []
        rule = []
        for a in range(len(self.rule)):
            rule = []
            rule.append(g)
            rule.append('rule ' + str(n))
            rule.append(str(self.rule[a]))
            rule.append(str(self.var[a]))
            rules.append(tuple(rule))
        return rules

def create_rule():
    return Rule()