# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:43:34 2019

@author: Venkatesh Ravichandran

Association Rule Learing - Apriori algorithm

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read csv data
df = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

#convert the dataset format to a list of lists which is accepted as input to apriori algorithm
transactions = []
for i in range(0,7501):
    transactions.append([str(df.values[i,j]) for j in range(0,20)]) 

#apriori algorithm to deduct the rules
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualisation of rules
results = list(rules)



