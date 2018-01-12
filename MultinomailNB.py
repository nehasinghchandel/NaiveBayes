# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:43:41 2018

@author: NC00486885
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances


data = pd.read_csv("D:/NC00486885/TECHM/Sports.csv",encoding='latin-1')
data.columns = ['CATEGORY','NEWS']
#print(data.head())
data.CATEGORY.unique()
data.groupby('CATEGORY').describe()
data['NUM_CATEGORY']=data.CATEGORY.map({'business':0,'sports':1})
#print(data.head())
x_train, x_test, y_train, y_test = train_test_split(data.NEWS, data.NUM_CATEGORY, random_state=42)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)
vect = CountVectorizer(ngram_range=(2,2))
#print(vect)
#converting traning features into numeric vector
X_train = vect.fit_transform(x_train)
#converting training labels into numeric vector
X_test = vect.transform(x_test)

mnb = MultinomialNB(alpha =1.0)

mnb.fit(X_train,y_train)

result= mnb.predict(X_test)
#print(result)


print(accuracy_score(result,y_test))
def predict_news(news):
    test = vect.transform(news)
    pred= mnb.predict(test)
    if pred  == 0:
         return 'Business'
    else:
         return 'Sports'
     
x=["the series would have been set up perfectly if India had come through today"]
r = predict_news(x)
print (r)
#print(accuracy_score(result,y_test))
str=''.join(x)
fl=vect.transform(x)
#print(confusion_matrix(y_test, result))
y_pred_prob = mnb.predict_proba(X_test)[:, 0]
print(y_pred_prob)
l=metrics.roc_auc_score(y_test, y_pred_prob)
print(l)
for f in X_test:
    euclidean_distances(fl, f) 
 