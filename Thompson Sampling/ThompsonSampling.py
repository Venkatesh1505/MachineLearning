# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 23:45:37 2019

@author: Venkatesh Ravichandran

Reinforcement  learning - Thompson sampling Algorithm

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read csv data
df = pd.read_csv('Ads_CTR_Optimisation.csv')

#Thompson sampling
import random
ads_selected = []
total_reward = 0
N = 10000
d = 10
number_of_random_1 = [0] * d
number_of_random_0 = [0] * d
for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_random_1[i] +1 , number_of_random_0[i]+1)
        if random_beta >max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        number_of_random_1[ad] += 1
    else:
        number_of_random_0[ad] += 1
    total_reward += reward    

#With thompson sampling, we get total reward around 2600 which is even greater than Upper confidence bound

#Visualisation
plt.hist(ads_selected)
plt.title('Thompson sampling')
plt.xlabel('Ads')
plt.ylabel('Click through rate')
plt.show()
