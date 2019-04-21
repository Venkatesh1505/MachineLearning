# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:21:05 2019

@author: saranya.ravichandran
"""

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import data
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#treat missing data --> not needed here
#treat categorical data --> not needed here
#split into train and test data --> not needed here as dataset has only 10 values.

#linear regression
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y)

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#plotting for linear regression
plt.scatter(X,y,color = 'red')
plt.plot(X,linear_reg.predict(X),color = 'blue')
plt.title('Truth or bluff (Linear regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#plotting for polynomial regression
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or bluff (polynomial regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


