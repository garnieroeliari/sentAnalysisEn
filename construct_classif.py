#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
# classify text as positive or negative
# train model from movie_reviews database in NLTK (there are 1000 positive and 1000 negative reviews in the movie_reviews data)
# use words as features 
# BAG OF WORDS Model 
# we use the top 5000 most common words as features
# convert words to features
# classify with naive bayes 
# save trained classifier with pickle

import nltk
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC


documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

# use words as features
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]
save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

# vectorizer : convert words to features
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_featuresets= open("pickled_algos/featuresets5k.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

# shuffle the featuresets to avoid biais
random.shuffle(featuresets)
print(len(featuresets))

# train classifier
training_set = featuresets[:1900]
# test classifier
testing_set = featuresets[1900:]

'''classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)'''

classifier = SklearnClassifier(LinearSVC())
classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)


# save the model (the trained classifier) with pickle
save_classifier = open("pickled_algos/LinearSVC5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

classifier_f = open("pickled_algos/LinearSVC5k.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats)
    
print(sentiment("this is a very enjoyable movie. you will leave it feeling good, and maybe thinking a bit as well. think dolphin tale. wish they had the ski jump sceens in imax"))
