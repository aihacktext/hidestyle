#!/usr/bin/env python3
#
# Webapp
#

from datetime import datetime
from bottle import route, run, view, post, request
from bottle import static_file, hook, response
import sys
import os

sys.path.append("../")
from text_processing.cleanse_text import normalize_text
from classification import classifier


clf = vectorizer = None


# Bottle routes
@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'

@route("/")
@view("index")
def index():
    return dict(orig_text='', anonymized=None, msg=None)

@view("index")
def receive_form_post():
    orig_text = request.forms.get('text')
    return dict(orig_text=orig_text, anonymized=mangled, msg=msg)

@route("/anonymize")
def anonymize():
    orig_text = request.query.text
    mangled = normalize_text(orig_text)
    anonymized, msg = classifier.classify_new_text(clf, vectorizer, mangled)
    hot_words = classifier.lime_explain(clf, vectorizer, mangled, num_examples_per_class=20,
                 num_lime_features=10)
    if hot_words:
        msg = "Better choose some synonims for: %s" % ' '.join(hot_words)
    else:
        msg = "No words to replace"
    return dict(orig_text=orig_text, anonymized=anonymized, msg=msg)

@route('/scripts/<filename>')
def server_static(filename):
    return static_file(filename, root='scripts/')

@post("/learn")
def learn():
    global clf, vectorizer
    max_items_per_author = int(request.forms.get('max_items_per_author'))
    clf, vectorizer = classifier.learn(
        max_items_per_author,
        num_lime_features=10,
        num_examples_per_class=max_items_per_author,
    )
    clf_fname, vectorizer_fname = classifier.generate_pkl_filenames()
    classifier.save_pickles(clf, clf_fname, vectorizer, vectorizer_fname)
    return """
<meta http-equiv="refresh" content="1;url=/" />
<p>Learning completed</p>
"""



def main():
    global clf, vectorizer

    try:
        clf, vectorizer = classifier.load_pickles()
        print("Existing pickles loaded")
    except Exception as e:
        print("Pickles not found, learning")
        learn()

    run(host='localhost', port=8080, debug=True, reloader=True)

if __name__ == "__main__":
    main()
