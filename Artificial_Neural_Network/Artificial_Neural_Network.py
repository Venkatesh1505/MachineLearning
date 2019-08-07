# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 21:34:18 2019

@author: Venkatesh Ravichandran

Artificial Neural Network

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

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
train_x = sc_x.fit_transform(train_x)
test_x = sc_x.transform(test_x)

# Building Artificial Neural Network(ANN model)
import keras
from keras.models import Sequential
from keras.layers import Dense

#Building the input and first hidden layer
classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(train_x, train_y,batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(test_x)
y_pred = y_pred>0.5

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_pred)









