# -*- coding: utf-8 -*-

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
df = pd.read_csv('Position_Salaries.csv')

#treat missing data --> not needed here
#treat categorical data --> not needed here

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#feature scaling
from sklearn.preprocessing import StandardScaler
x_sc = StandardScaler()
y_sc = StandardScaler()
X = x_sc.fit_transform(X)
y = y_sc.fit_transform(y)

#svr implementation
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') #using rbf(Gaussian kernel)
regressor.fit(X,y)


#predict output of new data
y_pred = y_sc.inverse_transform(regressor.predict(x_sc.transform(np.array([[6.5]]))))
#plot data
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('SVR regression')
plt.show()
