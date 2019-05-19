# -*- coding: utf-8 -*-
"""
Created on Sun May 19 18:11:38 2019

@author: Venkatesh Ravichandran
"""

#K-NN Classification

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
df = pd.read_csv('Social_Network_Ads.csv')
df = df.iloc[:,2:5]

#treat missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
#imputer.fit(df[:,1])
#df[:,1] = imputer.transform(df[:,1])

#treat categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder = LabelEncoder()
#df[:,1] = labelEncoder.fit_transform(df[:,1])
#onehotEncoder = OneHotEncoder(categorical_features = [1])
#df = onehotEncoder.fit_transform().to_array()

#split into train and test set
x = df.iloc[:,:2].values
y = df.iloc[:,2].values
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xtrain = scX.fit_transform(xtrain)
xtest = scX.fit_transform(xtest)

#K-NN Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(xtrain,ytrain)

#Predict for test set
y_pred = classifier.predict(xtest)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)

#Visualization using contourplot
from matplotlib.colors import ListedColormap

xset, yset = xtrain, ytrain

x1,x2 = np.meshgrid(np.arange(start = xset[:,0].min()-1, stop = xset[:,0].max()+1, step = 0.01),
                              np.arange(start = xset[:,1].min()-1, stop = xset[:,1].max()+1,step = 0.01))

arr = np.array([x1.ravel(),x2.ravel()])
arrT = arr.T
plt.contourf(x1, x2, classifier.predict(arrT).reshape(x1.shape),
                                        alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset==j,0],xset[yset==j, 1],
                c = ListedColormap(('red','green'))(i), label = j)
    
plt.title('K-NN Classification')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()