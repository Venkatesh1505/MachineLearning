# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:25:25 2019

@author: Venkatesh Ravichandran

Clustering - Hierarchical clustering

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read mall data csv
df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,[3,4]].values

#Dendrogram for finding the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()

#from the dendrogram, we find the optimal number of clusters as 5
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',linkage = 'ward')
y_sch = clustering.fit_predict(X)

#Visualization of clusters
plt.scatter(X[y_sch==0,0],X[y_sch==0,1],s=100,c='red',label = 'Careful')
plt.scatter(X[y_sch==1,0],X[y_sch==1,1],s=100,c='blue',label = 'Standard')
plt.scatter(X[y_sch==2,0],X[y_sch==2,1],s=100,c='green',label = 'arget')
plt.scatter(X[y_sch==3,0],X[y_sch==3,1],s=100,c='magenta',label = 'Careless')
plt.scatter(X[y_sch==4,0],X[y_sch==4,1],s=100,c='cyan',label = 'Sensible')
plt.title('Hierarchical Clustering')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()

