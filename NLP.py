# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:12:23 2020

@author: gokul.sangale
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import re

import nltk 

from nltk.corpus import stopwords
 
dataset=pd.read_csv('IMDB Dataset.csv');

corpus = []
for i in range(0,len(dataset)): 
    review = re.sub('[^a-zA-z]',' ', dataset['review'][i])
    review = review.lower()
    corpus.append(review)
    
y = pd.get_dummies(dataset['sentiment'],drop_first=True)




from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus).toarray()

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#fittin naive bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#prediction
y_pred=classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)