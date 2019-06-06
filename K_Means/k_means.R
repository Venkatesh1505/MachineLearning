
#read csv data
df <- read.csv('Mall_Customers.csv')
df = df[4:5]

#use elbow method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for(i in 1:10) wcss[i] <- sum(kmeans(df,i)$withinss)
plot(1:10,wcss,type='b', main = paste('Elbow method'),xlab = 'Number of clusters', ylab = 'WCSS')

#from plot, we get optimal number of clusters as 5
kMeans = kmeans(df,5,iter.max=10,nstart=1)

#Visualization of clusters
library(cluster)
clusplot(df, kMeans$cluster, lines = 0,shade =TRUE,
         color = TRUE, labels = 2, plotchar = FALSE,
         span = TRUE, main = paste('Cluster of clients'),
         xlab = 'Income',ylab='Purchase score')