#!/usr/bin/env python
#
# Text classifier
#
#
import gzip
from optparse import OptionParser
from contextlib import contextmanager
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from time import time
from sklearn import metrics
import os.path

import requests

from sklearn.externals import joblib

from sklearn.pipeline import Pipeline

from nltk.classify.scikitlearn import SklearnClassifier

classif = SklearnClassifier(LinearSVC())

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB


@contextmanager
def timer(before, after=None):
    before = before.capitalize()
    print "%s..." % before
    t = time()

    yield

    if after is None:
        after = "%s done" % before
    else:
        after = after.capitalize()

    print "%s in %.3fs" % (after, time() - t)

def download_if_needed(url, fname):
    if os.path.isfile(fname):
        return
    with timer('downloading %s' % url):
        with open(fname, 'w') as f:
            resp = requests.get(url, stream=True)
            f.write(resp.raw)

def extract_authors(corpus):
    return {i.split('-')[0] for i in corpus.fileids()}

def generate_pkl_filenames():
    return ('clf.pkl', 'vect.pkl')

def parse_args():
    op = OptionParser()
    op.add_option('--learn', action='store_true')
    opts, args = op.parse_args()
    return opts

def show_most_informative_features(vectorizer, clf, n=20):
    print 'Most informative features:'
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

    print '--'

def learn():
    """Learn to classify texts by author
    """
    X = []
    y = []

    with timer('parsing imdb'):
        with open('imdb62.txt') as f:
            cnt = 0
            for line in f:
                tok = line.split(None, 5)
                author = tok[1]
                body = tok[5]
                X.append(body)
                y.append(author)

    pipeline = Pipeline([
        ('tfidf', TfidfTransformer()),
        ('chi2', SelectKBest(chi2, k=1000)),
        ('nb', MultinomialNB())
    ])
    #classif = SklearnClassifier(pipeline)
    #pipeline = Pipeline([
    #    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    #    ('classifier',         MultinomialNB()),
    #])
    #pipeline.fit(X_train, y_train)
    #print pipeline.score(X_test, y_test)

    Xo = X
    yo = y
    with timer('fitting', 'fitting done'):
        vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
        )
        X = vectorizer.fit_transform(X)


    with timer('splitting'):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42)

    print("total: n_samples: %d, n_features: %d" % X.shape)
    print("train: n_samples: %d, n_features: %d" % X_train.shape)
    print("test:  n_samples: %d, n_features: %d" % X_test.shape)

    #clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
    clf = LinearSVC()

    with timer('training'):
        clf.fit(X_train, y_train)

    with timer('scoring'):
        score = clf.score(X_test, y_test)
        print("Score:   %0.3f" % score)

    show_most_informative_features(vectorizer, clf, n=45)
    print(clf.decision_function(X_train))

    #with timer('Measuring confidence'):
    #    decision = clf.decision_function(X_train)
    #    clf._max_confidence = max(decision)
    #    clf._min_confidence = min(decision)
    clf._max_confidence = 1.0
    clf._min_confidence = 0.0


    return clf, vectorizer

def main():

    opts = parse_args()

    download_if_needed(
        "https://doc-10-5g-docs.googleusercontent.com/docs/securesc/458l16557e3dc79l5g0asf6656vqredc/t3bttbjtqb4r1cari89roveb8c4nns7t/1487426400000/07115181027429506682/12988091470430942163/0B3emjZ5O5vDtQXdRSS04REZXYmM?e=download",
        "imdb62.txt"
    )

    clf_fname, vectorizer_fname = generate_pkl_filenames()
    if opts.learn or not os.path.isfile(clf_fname):
        clf, vectorizer = learn()
        with timer('Saving pickles'):
            joblib.dump(clf, clf_fname)
            joblib.dump(vectorizer, vectorizer_fname)

    else:
        with timer('Loading pickles'):
            clf = joblib.load(clf_fname)
            vectorizer = joblib.load(vectorizer_fname)



    delta = abs(clf._max_confidence - clf._min_confidence)

    with timer('Classifying new author'):
        with open('new_text') as f:
            new_body = f.read()

        X = vectorizer.transform([new_body, ])
        dfu = clf.decision_function(X)
        print("decision function: %s" % dfu)
        predicted = clf.decision_function(X)[0]
        p_min = min(predicted)
        p_max = max(predicted)
        delta = p_max - p_min
        for p, name in zip(predicted, clf.classes_):
            print("%f %s" % ((p - p_min) / delta, name))


if __name__ == '__main__':
    main()
