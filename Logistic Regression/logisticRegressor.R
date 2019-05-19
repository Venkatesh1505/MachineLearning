# read data
df = read.csv('Social_Network_Ads.csv')
#trim dataset to needed data
df = df[3:5]

#treat missing data
#df$EstimatedSalary = ifelse(is.na(df$EstimatedSalary),
 #                           ave(df$EstimatedSalary, FUN = function(x) mean(df$EstimatedSalary,rm.na = TRUE)),
  #                          df$EstimatedSalary)

#treat categorical data
df$Purchased = factor(df$Purchased,levels = c(0,1))

#split into training and test data
library(caTools)
split = sample.split(df$Purchased, SplitRatio = 0.75)
train = subset(df, split == TRUE)
test = subset(df, split == FALSE)

#feature scaling
train[1:2] = scale(train[1:2])
test[1:2] = scale(test[1:2])

#logistic Regression - Building model
classifier = glm(formula = Purchased ~ ., family='binomial', data = train)

#predicting values for test data
pred_prob = predict(classifier, type = 'response', newdata = test[1:2])
y_pred = ifelse(pred_prob > 0.5 , 1, 0)

#confusion matrix
cm = table(test[,3],y_pred)

#Visualization using contourplot
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test
x1 = seq(min(set[,1])-1,max(set[,1])+1,by = 0.01)
x2 = seq(min(set[,2])-1,max(set[,2])+1,by = 0.01)
grid_set = expand.grid(x1,x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[,-3], 
     main = 'Logistic Regression plot', 
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(x1), ylim = range(x2))
contour(x1,x2, matrix(as.numeric(y_grid),length(x1),length(x2)),add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1,'springgreen3','tomato'))
points(set, pch = 21, bg = ifelse(set[,3]==1, 'green4','red3'))