
df = read.csv("C:\\Users\\saranya.ravichandran\\Desktop\\venky\\Tech\\DS\\ML-A-Z\\Machine Learning A-Z\\Part 1 - Data Preprocessing\\Data_Preprocessing\\Data.csv")

df$Age = ifelse(is.na(df$Age),
                ave(df$Age,FUN = function(x) mean(x,na.rm = TRUE)),
                df$Age)


df$Salary = ifelse(is.na(df$Salary),
                ave(df$Salary,FUN = function(x) mean(x,na.rm = TRUE)),
                df$Salary)
df$Country = factor(df$Country,
                    levels = c('France','Spain','Germany'),
                    labels = c(1,2,3)
                    )
df$Purchased = factor(df$Purchased,
                    levels = c('No','Yes'),
                    labels = c(0,1)
                    )
#install.packages('caTools')
library(caTools)
split = sample.split(df$Purchased,SplitRatio = 0.8)
trainset = subset(df,split==TRUE)
testset = subset(df,split == FALSE)

#feature scaling

trainset[,2:3] = scale(trainset[,2:3])
testset[,2:3] = scale(testset[,2:3])


