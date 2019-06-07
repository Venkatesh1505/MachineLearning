
# read csv data
df <- read.csv('Market_Basket_Optimisation.csv', header = FALSE)
#install.packages('arules')
library(arules)
#For Eclat - We convert the dataset to sparse matrix(Matrix which contains 0s and 1s where 0s are more)
df <- read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(df)
#to plot the frequency of all the items purchased
itemFrequencyPlot(df, topN = 100)

#Eclat algorithm
rules = eclat(df, parameter = list(support = 0.004,minlen = 2))

#displaying the rules
inspect(sort(rules, by = 'support')[1:10])
