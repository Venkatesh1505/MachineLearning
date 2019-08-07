#Artificial Neural Networks

#read csv data
df = read.csv("Churn_Modelling.csv")
df = df[4:14]

#mapping categorical variables
df$Geography = as.numeric(factor(df$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
df$Gender = as.numeric(factor(df$Gender,
                              levels = c('Male','Female'),
                              labels = c(1, 2)))

#splitting into train and test sets
library(caTools)
set.seed(100)
split = sample.split(df$Exited, SplitRatio = 0.8)
train = subset(df, split == TRUE)
test = subset(df, split == FALSE)

#Feature scaling
train[-11] = scale(train[-11])
test[-11] = scale(test[-11])

#Building ANN
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)

classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(train),
                              activation = 'Rectifier',
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2)

#prediction on new data
y_pred = h2o.predict(classifier, newdata = as.h2o(test[-11]))

y_pred = (y_pred>0.5)

y_pred = as.vector(y_pred)

#Confusion matrix
cm = table(test[,11], y_pred)

#shutdown the h2o
h2o.shutdown()