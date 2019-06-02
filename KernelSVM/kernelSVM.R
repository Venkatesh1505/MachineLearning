#import data csv
df = read.csv('Social_Network_Ads.csv')
df = df[3:5]

#Data preprocessing
#treat missing data
# df$Age = ifelse(is.na(df$Age),
#                 ave(df$Age, FUN = function(x) mean(x,na.rm = TRUE)),
#               df$Age)

#treat categorical data
df$Purchased = factor(df$Purchased,levels = c(0,1))

#split into train and test set
library(caTools)
set.seed(123)
split = sample.split(df$Purchased, SplitRatio = 0.8)
train = subset(df,split == TRUE)
test = subset(df, split == FALSE)

#Feature scaling
train[-3] = scale(train[-3])
test[-3] = scale(test[-3])

#Kernel SVM classification
library(e1071)
classifier = svm(formula = Purchased ~., 
                 data = train,
                 type = 'C-classification',
                 kernel = 'radial')

#predict test data
y_pred = predict(classifier, newdata = test[-3])

#Confusion matrix
cm = table(test[,3], y_pred)

#Contour plot for Visualization
library(ElemStatLearn)
set = test
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'Classifier (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))