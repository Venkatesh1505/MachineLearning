# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 21:57:49 2019

@author: Venkatesh Ravichandran

Clustering Algorithm - K-Means

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read csv file
df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,3:5].values

#Build the K-Means Model
from sklearn.cluster import KMeans
#selecting number of clusters using elbow method
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init='k-means++',n_init=10,max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('Elbow method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()

#From Elbow method, we found that Ideal number of clusters is 5
noOfClusters = 5
kmeansModel = KMeans(n_clusters = noOfClusters, init = 'k-means++', n_init = 10, max_iter = 300, random_state=0)
y_cluster = kmeansModel.fit_predict(X)

plt.scatter(X[y_cluster==1,0],X[y_cluster==1,1],s = 100,color='red',label='Target')
plt.scatter(X[y_cluster==2,0],X[y_cluster==2,1],s = 100,color='blue',label='Rich')
plt.scatter(X[y_cluster==3,0],X[y_cluster==3,1],s = 100,color='green',label='Careless')
plt.scatter(X[y_cluster==4,0],X[y_cluster==4,1],s = 100,color='cyan',label='Sensible')
plt.scatter(X[y_cluster==0,0],X[y_cluster==0,1],s = 100,color='magenta',label='Stingy')
plt.scatter(kmeansModel.cluster_centers_[:,0],kmeansModel.cluster_centers_[:,1],s=300,color='red',label='Centroid')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.legend()
plt.show()



