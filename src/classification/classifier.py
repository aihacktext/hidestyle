#!/usr/bin/env python3
#
# Text classifier
#

from contextlib import contextmanager
from lime.lime_text import LimeTextExplainer
from optparse import OptionParser
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import DistanceMetric
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from time import time
import numpy as np
import os.path
import scipy
import requests

@contextmanager
def timer(before, after=None):
    before = before.capitalize()
    print("%s..." % before)
    t = time()

    yield

    if after is None:
        after =("%s done" % before)
    else:
        after = after.capitalize()

    print("%s in %.3fs" % (after, time() - t))

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
    print('Most informative features:')
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

    print("--")

def learn():
    """Learn to classify texts by author
    """
    X = []
    y = []

    seen_authors = {}
    with timer('parsing imdb'):
        with open('imdb62.txt') as f:
            for line in f:
                tok = line.split(None, 5)
                author = tok[1]
                body = tok[5]
                try:
                    if seen_authors[author] == 50:
                        continue
                    seen_authors[author] += 1
                except KeyError:
                    seen_authors[author] = 1

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
        #vectorizer = CountVectorizer(
        #    analyzer='word',
        #    ngram_range=(1, 1),
        #    max_features=10000,
        #    stop_words='english'
        #)
        vectorizer = TfidfVectorizer(
            stop_words='english',
            #ngram_range=(1, 3),
            lowercase=True,
        )
        X = vectorizer.fit_transform(X)


    with timer('splitting'):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.10, random_state=42)

    print("total: n_samples: %d, n_features: %d" % X.shape)
    print("train: n_samples: %d, n_features: %d" % X_train.shape)
    print("test:  n_samples: %d, n_features: %d" % X_test.shape)

    #clf = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)
    clf = LinearSVC()
    clf = LogisticRegression()

    with timer('training'):
        clf.fit(X_train, y_train)

    with timer('scoring'):
        score = clf.score(X_test, y_test)
        print("Score:   %0.3f" % score)

    show_most_informative_features(vectorizer, clf, n=45)
    print(clf.decision_function(X_train))

    num_examples_per_class = 50
    user_input = "..................."
    user_features = vectorizer.transform([user_input])
    duplicated_user_features = scipy.sparse.vstack([user_features for _ in range(int(num_examples_per_class / user_features.shape[0]))])

    rand_review_features = X
    print('Review features shape:', rand_review_features.shape)
    return clf, vectorizer

def save_pickles(clf, clf_fname, vectorizer, vectorizer_fname):
    with timer('Saving pickles'):
        joblib.dump(clf, clf_fname)
        joblib.dump(vectorizer, vectorizer_fname)

def load_pickles(clf_fname=None, vectorizer_fname=None):
    if clf_fname == None:
        clf_fname, vectorizer_fname = generate_pkl_filenames()

    with timer('Loading pickles'):
        clf = joblib.load(clf_fname)
        vectorizer = joblib.load(vectorizer_fname)
        return clf, vectorizer

def classify_new_text(clf, vectorizer, new_body):
    """Classify text
    returns anonymized text, help message
    """

    with timer('Classifying new author'):

        X = vectorizer.transform([new_body, ])
        dfu = clf.decision_function(X)
        print("decision function: %s" % dfu)
        predicted = dfu[0]
        p_min = min(predicted)
        p_max = max(predicted)
        delta = p_max - p_min
        for p, name in zip(predicted, clf.classes_):
            print("%f %s" % ((p - p_min) / delta, name))

    dist = DistanceMetric.get_metric('euclidean')
    avg = sum(predicted) / len(predicted)

    jd = np.ones((1, len(predicted))) * avg
    d = dist.pairwise(dfu, jd)[0][0]

    jd = np.ones((1, len(predicted)))
    d_one = dist.pairwise(dfu, jd)[0][0]

    jd = np.ones((1, len(predicted))) * 0.0
    d_zero = dist.pairwise(dfu, jd)[0][0]

    return new_body, "Distances: %f %f" % (d_zero, d_one)


def main():

    opts = parse_args()

    download_if_needed(
        "https://doc-10-5g-docs.googleusercontent.com/docs/securesc/458l16557e3dc79l5g0asf6656vqredc/t3bttbjtqb4r1cari89roveb8c4nns7t/1487426400000/07115181027429506682/12988091470430942163/0B3emjZ5O5vDtQXdRSS04REZXYmM?e=download",
        "imdb62.txt"
    )

    clf_fname, vectorizer_fname = generate_pkl_filenames()
    if opts.learn or not os.path.isfile(clf_fname):
        clf, vectorizer = learn()
        save_pickles(clf, clf_fname, vectorizer, vectorizer_fname)

    else:
        clf, vectorizer = load_pickles(clf_fname, vectorizer_fname)

    with open('new_text') as f:
        new_body = f.read()
    classify_new_text(clf, vectorizer, new_body)


if __name__ == '__main__':
    main()
