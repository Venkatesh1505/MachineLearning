# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:24:04 2019

@author: Venkatesh Ravichandran

Implementation of Kernel SVM(Gaussian Kernel) for Social Network Ads Classification

"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data csv
df = pd.read_csv("Social_Network_Ads.csv")
X = df.iloc[:,2:4].values
y = df.iloc[:,4].values

#Data preprocessing
#treat missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = np.NaN, strategy = 'mean',axis = 0)
#imputer = imputer.fit(X[:,0])
#X[:,0] = imputer.transform(X[:,0])

#treat categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder = LabelEncoder()
#X[:,1] = labelEncoder.fit_transform(X[:,1])
#oneHotEncoder = OneHotEncoder(categorical_features = [1])
#X = oneHotEncoder.fit_transform(X).to_array()

#split into train and test set
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaleX = StandardScaler()
xtrain = scaleX.fit_transform(xtrain)
xtest = scaleX.fit_transform(xtest)

#Kernel SVM Classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state=0)
classifier.fit(xtrain,ytrain)

#Predict for test data
y_pred = classifier.predict(xtest)

#Confusion matrix to check the accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,ytest)

#Contour plot for visualization
from matplotlib.colors import ListedColormap
xset, yset = xtest,ytest
X1,X2 = np.meshgrid(np.arange(start = xset[:,0].min()-1, stop = xset[:,0].max()+1, step = 0.01),
                    np.arange(start = xset[:,1].min()-1, stop = xset[:,1].max()+1, step = 0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j, 0], xset[yset == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()