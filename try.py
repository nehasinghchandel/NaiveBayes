# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:16:30 2018

@author: NC00486885
"""
import collections, re
import csv
from pprint import pprint
from nltk import NaiveBayesClassifier

with open('D:/NC00486885/TECHM/MNaive_Bayes.csv', newline='') as file:
    reader = csv.reader(file)
    l = list(map(tuple, reader))
#pprint(l)
def gender_features(word):
    return {'last_letter': word[-1]}
#print(gender_features('Gary'))
featuresets = [(gender_features(n), g) for (n, g) in l]
#print(featuresets)
#print(len(featuresets))
train_set, test_set = featuresets[:30], featuresets[30:]
#print(len(train_set))
nb_classifier = NaiveBayesClassifier.train(train_set)
print(nb_classifier.classify(gender_features('Deepti')))
#print(nb_classifier.classify(gender_features('Grace')))
#print(nb_classifier.show_most_informative_features(5))