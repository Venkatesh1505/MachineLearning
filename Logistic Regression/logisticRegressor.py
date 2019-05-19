# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:10:26 2019

@author: Venkatesh Ravichandran

Code to implement Logistic regression
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
df = pd.read_csv('Social_Network_Ads.csv')

X = df.iloc[:,2:4].values
y = df.iloc[:,4].values

#treat missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:,2:4])
#X[:,2:4] = imputer.transform(X[:,2:4])

#treat categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#xencoder = LabelEncoder()
#X[:,1] = xencoder.fit_transform(X[:,1])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform().to_array()

#split into train and test sets
from sklearn.cross_validation import train_test_split
x_train,x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)

#fit the model
classifier.fit(x_train, y_train)

#predict from the model
y_pred = classifier.predict(x_test)

#confusion matrix - to check the accuracy of the model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#contour plot to visualize the prediction
from matplotlib.colors import ListedColormap
xset, yset = x_train, y_train

X1, X2 = np.meshgrid(np.arange(xset[:,0].min()-1,xset[:,0].max()+1,0.01),
                     np.arange(xset[:,1].min()-1,xset[:,1].max()+1,0.01)
                     )
arr = np.array([X1.ravel(),X2.ravel()])
arrT = arr.T
plt.contourf(X1,X2,classifier.predict(arrT).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(),X2.max())

for i, j in enumerate(np.unique(yset)):
    if i == 1 and j == 1:
        print(i,j)
    plt.scatter(xset[yset==j,0],xset[yset==j,1], c = ListedColormap(('red','green'))(i),
                label = j)

plt.title('Logistic Regression')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()




