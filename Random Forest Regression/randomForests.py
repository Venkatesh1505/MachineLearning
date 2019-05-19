#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('C:\\Users\\saranya.ravichandran\\Desktop\\venky\\Tech\\DS\\ML-A-Z\\Machine Learning A-Z\\Part 2 - Regression\\Section 8 - Decision Tree Regression\\Position_Salaries.csv')

#treat missing data --> not needed here
#from sklearn.preprocessing import Imputer

#treat categorical data --> not needed here
#from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#split data into dependant and independant
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#feature scaling --> not needed here
#from sklearn.preprocessing import StandardScaler

#Random forest implementation
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

y_pred = regressor.predict(6.5)

#plot data
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.title('Random forest regression plot')
plt.show()