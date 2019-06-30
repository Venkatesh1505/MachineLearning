# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 17:09:17 2019

@author: Venkatesh Ravichandran

Natural Language Processing

"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Now stemming is done. Then we have to create the bag of words model(Sparse matrix)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,1].values

#Bag of Words model is built. Now this is similar to classification problem
from sklearn.cross_validation import train_test_split
trainX, testX, trainY, testY = train_test_split(X, y, random_state=0,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(trainX,trainY)
y_pred = classifier.predict(testX)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testY, y_pred)
print('Accuracy is : '+str((55+91)/200))

