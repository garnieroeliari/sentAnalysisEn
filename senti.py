# classify tweet 
# use pickled model, featuresets, word features and documents saved from construct_classif.py

import nltk
import random
import pickle
from nltk.tokenize import word_tokenize



def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
    
def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats)
    
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print(sentiment("very good"))
