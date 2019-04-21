# import data

df = read.csv('Position_Salaries.csv')
#index starts at 1 for R
df = df[2:3]

#treat missing data -->not needed here
#treat categorical data --> not needed here
#split into train and test data --> not needed here as dataset is small
#feature scaling --> not needed here

#linear regression
lin_reg = lm(formula = Salary ~ ., data = df)
#install.packages("ggplot2")
#plot data
library(ggplot2)
ggplot() +
  geom_point(aes(x = df$Level, y = df$Salary),color = 'red') +
  geom_line(aes(x = df$Level, y = predict(lin_reg,newdata = df)),color = 'blue') +
  ggtitle('Linear regression(Level Vs Salary') +
  xlab('Level') +
  ylab('Salary')

#polynomial regression
df$Level1 = df$Level^2
df$Level2 = df$Level^3
df$Level3 = df$Level^4

poly_reg = lm(formula = Salary ~ ., data = df)
#install.packages("ggplot2")
#plot data
library(ggplot2)
ggplot() +
  geom_point(aes(x = df$Level, y = df$Salary),color = 'red') +
  geom_line(aes(x = df$Level, y = predict(poly_reg,newdata = df)),color = 'blue') +
  ggtitle('Polynomial regression(Level Vs Salary)') +
  xlab('Level') +
  ylab('Salary')

#predicting new results with polynomial regression
y_predict = predict(poly_reg,data.frame(Level = 6.5, Level1 = 6.5^2,
                                      Level2 = 6.5^3, Level3 = 6.5^4))