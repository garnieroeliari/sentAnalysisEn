#!/usr/bin/python3.4
# -*- coding: utf-8 -*-
# classify text as positive or negative
# train model from movie_reviews database in NLTK
# use words as features 
# BAG OF WORDS Model 
# we use the top 3000 most common words as features
# convert words to features
# classify with naive bayes 
# save trained classifier with pickle

import nltk
import random
import pickle
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)


all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)


word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)

'''
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()'''

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

text = "very good"
text2= "Inspiring story. A bit hollywood'ish. Moves right along with good acting. Our entire family enjoyed it. "
text3 = "This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"
text4 = "this is a very enjoyable movie. you will leave it feeling good, and maybe thinking a bit as well. think dolphin tale. wish they had the ski jump sceens in imax"

print(classifier.classify(find_features(text)))
print(classifier.classify(find_features(text2)))
print(classifier.classify(find_features(text3)))
print(classifier.classify(find_features(text4)))

'''def sentiment(text):
    feats = find_features(text)
    return classifier.classify(feats)

print(sentiment("very good"))
print(sentiment("Inspiring story. A bit hollywood'ish. Moves right along with good acting. Our entire family enjoyed it. "))
print(sentiment("This movie was awesome! The acting was great, plot was wonderful, and there were pythons...so yea!"))
print(sentiment("this is a very enjoyable movie. you will leave it feeling good, and maybe thinking a bit as well. think dolphin tale. wish they had the ski jump sceens in imax"))'''
