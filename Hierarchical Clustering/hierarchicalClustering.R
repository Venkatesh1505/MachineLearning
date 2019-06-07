#read mall csv data
df <- read.csv('Mall_Customers.csv')
X = df[4:5]

#Dendrogram to find the optimal number of clusters
dendrogram = hclust(dist(X,method='euclidean'),method='ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distance')

#From dendrogram, we found the optimal number of clusters as 5

#Fit the model
hclus = hclust(dist(X,method='euclidean'),method='ward.D')
y_hc = cutree(hclus,5)

#Visualization of clusters
library(cluster)
clusplot(X, y_hc, lines = 0,shade =TRUE,
         color = TRUE, labels = 2, plotchar = FALSE,
         span = TRUE, main = paste('Cluster of clients'),
         xlab = 'Income',ylab='Purchase score')


