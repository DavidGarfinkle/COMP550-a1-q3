###
# train_models.py
#
# use bags of words from mongodb and train models with it

import pymongo
client = pymongo.MongoClient("localhost", 8000)
db = client.sentiment_analysis
db_collection = db.bags_of_words

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from itertools import product, chain

# Load ground truth array
with open('data/db/ground_truth.npy', 'rb') as f:
    y = np.load(f)

results = pd.DataFrame()

# Classify with the variety of bags
for bag in db_collection.find({ 'classifier': { '$exists': False } }):
    with open('data/db/{}.npy'.format(bag['_id']), 'rb') as f:
        X = np.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    for (classifier, hyperparam_name, hyperparam_val) in chain(
            #((GaussianNB, '', None),),
            product((BernoulliNB,), ('alpha',), np.linspace(0.1, 1.9, 10)),
            product((LogisticRegression,), ('C',), np.linspace(0.1, 1.9, 10)),
            product((LinearSVC,), ('C',), np.linspace(0.1, 1.9, 10))):
        print("training classifier {} with hyperparam {} {}".format(classifier.__name__, hyperparam_name, hyperparam_val))
        model = classifier(**{hyperparam_name: hyperparam_val})
        model.fit(X_train, y_train)

        print(classifier.__name__)
        print(bag)
        print(classification_report(y_test, model.predict(X_test)))

        bag['accuracy'] = accuracy_score(y_test, model.predict(X_test))
        bag['classifier'] = classifier.__name__
        bag['hyperparameter'] = "{}: {}".format(hyperparam_name, hyperparam_val)
        results = results.append(bag, ignore_index=True)

    results.to_pickle('data/db/results.pkl')


