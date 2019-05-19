#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
df = pd.read_csv('C:\\Users\\saranya.ravichandran\\Desktop\\venky\\Tech\\DS\\ML-A-Z\\Machine Learning A-Z\\Part 2 - Regression\\Section 8 - Decision Tree Regression\\Position_Salaries.csv')

#treat missing data --> no missing data here
#treat categorical data --> no categorical data here

X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#feature scaling

#decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)


#predict new data

y_pred = regressor.predict(6.5)

#plot values
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')

plt.title('Decision tree regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
