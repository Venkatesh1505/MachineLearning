#Natural Language Processing
#read data
df = read.delim('Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

#clean the data - Stemming
install.packages('tm')
install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(df$Review))
#to convert all the characters to lowercase
corpus = tm_map(corpus,content_transformer(tolower))
#to remove all the punctuation
corpus = tm_map(corpus, removePunctuation)
#to remove the numbers
corpus = tm_map(corpus, removeNumbers)
#to remove the stopwords
corpus = tm_map(corpus, removeWords, stopwords())
#to stem the words(convert everything to rootwords)
corpus = tm_map(corpus, stemDocument)
#to remove the whitespaces
corpus = tm_map(corpus, stripWhitespace)

#Cleaning is done. Now create the bag of words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.array(dtm))
dataset$Liked = df$Liked


#Using classification algorithm to predict

dataset$Liked = factor(dataset$Liked, levels = c(0, 1))
#random forest
library(caTools)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
train = subset(dataset, split == TRUE)
test = subset(dataset, split == FALSE)

library(randomForest)
classifier = randomForest(x = train[-692], y = train$Liked, ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test[-692])

# Making the Confusion Matrix
cm = table(test[, 692], y_pred)