# import dataset

df = read.csv('50_Startups.csv')

#treat missing data --> here no missing data
# treat categorical data

df$State = factor(df$State, levels = c('New York', 'California', 'Florida'),
                  labels = c(1, 2, 3)
                  )

#split into training and test set

library(caTools)
split = sample.split(df$Profit,SplitRatio = 0.8)
train = subset(df, split == TRUE)
test = subset(df, split == FALSE)

#Feature scaling --> not needed for this dataset
# linear regression

regressor = lm(formula = Profit ~. , data = train)
y_predict = predict(regressor, newdata = test)
#to get the p value of all the predictors
summary(regressor)

#backward elimination
#build model with all the predictors
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = df)
summary(regressor)
#remove the predictor with highest p value --> p value is greater than Significance level
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = df)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = df)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, data = df)
summary(regressor)
