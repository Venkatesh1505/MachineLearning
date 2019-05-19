#import data
df = read.csv('Position_Salaries.csv')
df = df[2:3]
#treat missing data -->not needed here

#treat categorical data --> not needed here

#split into train and test set --> not needed here
set.seed(1234)
#build Random forest regression model
#install.packages('randomForest')
library(randomForest)

regressor = randomForest(x = df[1],y = df$Salary, ntree = 500)

#predict new data
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))

#plot values
xgrid = seq(min(df$Level),max(df$Level),0.01)
library(ggplot2)
ggplot()+
  geom_point(aes(x=df$Level,y=df$Salary),color='red')+
  geom_line(aes(x=xgrid,y=predict(regressor,newdata = data.frame(Level = xgrid))),color='blue')+
  xlab('Position')+
  ylab('Salary')+
  ggtitle('Random forest regression')