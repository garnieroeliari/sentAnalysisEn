# classify tweet 
# use pickled model, featuresets, word features and documents saved from construct_classif.py

import nltk
import random
import pickle
from nltk.tokenize import word_tokenize


documents_f = open("pickled_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


word_features5k_f = open("pickled_algos/word_features3k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets_f = open("pickled_algos/featuresets3k.pickle", "rb")
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[1900:]
training_set = featuresets[:1900]


open_file = open("pickled_algos/naivebayes3k.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats)