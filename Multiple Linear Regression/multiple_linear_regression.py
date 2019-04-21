# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:27:25 2019

@author: saranya.ravichandran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('50_Startups.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
df.describe()
#from describe, we can see that there are no missing data, so skip treating missing data
#next is treating categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
X[:,3] = labelEncoderX.fit_transform(X[:,3])
onehotEncoderX = OneHotEncoder(categorical_features=[3])
X = onehotEncoderX.fit_transform(X).toarray()
#So we have treated the categorical data using LabelEncoder and OneHotEncoder.
#To avoid dummy variable trap, always remove one variable from the n dummy variables created)
X = X[:,1:]
#Next step is splitting data into training data and test data.
from sklearn.cross_validation import train_test_split
[trainX, testX, trainY, testY] = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Next step is feature scaling which is done using StandardScaler. Not needed here
#So now build linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(trainX,trainY)
predictions = regressor.predict(testX)
#now predictions are done. Now we are not sure if this is the optimal model with all the variables
#contributing to the model.So we build optimal model
#backward elimination - we use statsmodels library for that
import statsmodels.formula.api as sm
#for statsmodels , we need to add 1 before X since it does not take care of constant factor
#in the regression formula theta0
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis=1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

