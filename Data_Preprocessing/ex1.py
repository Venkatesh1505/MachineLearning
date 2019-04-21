# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 22:56:08 2019

@author: saranya.ravichandran
"""
#importing libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("C:\\Users\\saranya.ravichandran\\Desktop\\venky\\Tech\\DS\\ML-A-Z\\Machine Learning A-Z\\Part 1 - Data Preprocessing\\Data_Preprocessing\\Data.csv")

x = df.iloc[:,:-1].values
           
y = df.iloc[:,3].values
         
#missing values
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = np.nan,strategy = 'mean',axis = 0)

imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#encoding categorical variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
#using labelencoder can be confusin to the system because it takes one as greater and other smaller
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#spliting train and test set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)

#feature scaling

from sklearn.preprocessing import StandardScaler
scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)

