#import data
df = read.csv('Position_Salaries.csv')

#treat missing data --> not needed here
#treat categorical data --> not needed here
df = df[2:3]

#Decision tree regression
#install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~., data = df, control = rpart.control(minsplit = 1))

#predict new data
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))
#plot values
#Decision tree is a non-continuous model. So feature scaling cannot affect the model performance in any way.
#For non-continuous models, visualization is done by splitting into small grids

x_grid = seq(min(df$Level),max(df$Level),0.01)
library(ggplot2)
ggplot()+
  geom_point(aes(x=df$Level,y=df$Salary),colour='red')+
  geom_line(aes(x=x_grid,y=predict(regressor,newdata = data.frame(Level = x_grid))))+
  ggtitle('Decision tree regression')+
  xlab('position')+
  ylab('Salary')

#random forest is a team of many decision trees which can be used for accurate results
