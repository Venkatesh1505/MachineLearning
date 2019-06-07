
# read csv data
df <- read.csv('Market_Basket_Optimisation.csv', header = FALSE)
#install.packages('arules')
library(arules)
#For Apriori - We convert the dataset to sparse matrix(Matrix which contains 0s and 1s where 0s are more)
df <- read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(df)
#to plot the frequency of all the items purchased
itemFrequencyPlot(df, topN = 100)

#Apriori algorithm
rules = apriori(df, parameter = list(support = 0.004,confidence = 0.2))

#displaying the rules
inspect(sort(rules, by = 'lift')[1:10])
