# read dataset
df = read.csv('Position_Salaries.csv')

#treat missing data --> not needed here
#treat categorical data --> not needed here
df = df[2:3]

#split into training and testing data --> not needed since dataset size is small
#SVR
#install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary~., data = df, type = 'eps-regression')

#predict new data
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

#plot values
library(ggplot2)
ggplot()+
  geom_point(aes(x = df$Level, y = df$Salary),colour='red')+
  geom_line(aes(x = df$Level, y = predict(regressor,newdata = df)),colour = 'blue')+
  ggtitle('SVR visualization')+
  xlab('Position')+
  ylab('Salary')
