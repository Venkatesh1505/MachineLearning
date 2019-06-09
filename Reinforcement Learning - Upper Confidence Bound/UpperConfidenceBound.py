# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:50:35 2019

@author: Venkatesh Ravichandran

Reinforcement Learning - Upper Confidence Bound Algorithm

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read csv data
df = pd.read_csv('Ads_CTR_Optimisation.csv')

#Using random selection to compare the result with Upper bound confidence algorithm
import random
d = 10
N = 10000
tot_reward = 0
for n in range(0, N):
    ad_selected = random.randrange(0,d)
    reward = df.values[n, ad_selected]
    tot_reward += reward
tot_reward = int(tot_reward)
        
#Upper confidence bound algorithm
import math
d = 10
sum_of_reward = [0]*d
number_of_selection = [0]*d
N = 10000
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if number_of_selection[i] > 0:
            avg_reward = sum_of_reward[i] / number_of_selection[i]
            delta = math.sqrt(3/2 * math.log(n + 1) / number_of_selection[i])
            upper_bound = avg_reward + delta
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selection[ad] += 1
    reward = df.values[n,ad]
    sum_of_reward[ad] += reward
    total_reward += reward
    
total_reward = int(total_reward)

#In random_Selection, we got total reward around 1200.
#But with Upper confidence bound, we get total reward of 2178 which is almost double of it.
#Hence it has done a good job

#Visualization of the results
plt.hist(ads_selected)
plt.title('Upper confidence bound - Number of times each ad is selected by the user')
plt.xlabel('Ads')
plt.ylabel('Number of times selected')
plt.show()

    


