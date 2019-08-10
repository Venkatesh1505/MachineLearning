# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:55:03 2019

@author: Venkatesh Ravichandran

XGBoost Implementation

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
df = pd.read_csv('Churn_Modelling.csv')

#assign dependent and independent variables
X = df.iloc[:, 3:13].values
y = df.iloc[:,13].values

#Treating categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder1 = LabelEncoder()
X[:,1] = labelEncoder1.fit_transform(X[:,1])
labelEncoder2 = LabelEncoder()
X[:,2] = labelEncoder2.fit_transform(X[:,2])

onehotEncoder = OneHotEncoder(categorical_features = [1])
X = onehotEncoder.fit_transform(X).toarray()
X = X[:,1:]

#splitting into train and test set
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

#XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(train_x, train_y)

y_pred = classifier.predict(test_x)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv = 10)
accuracies.mean()
accuracies.std()

