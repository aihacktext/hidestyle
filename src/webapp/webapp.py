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

example_text = """
Chuck Versus the Nemesis is the funniest action-packed episode of " Chuck " yet !	This episode takes place on Thanksgiving and Black Friday the next day so I'll just mention some tidbits here : With Bryce now revealed to be alive , Chuck gets in the interrogation room with him and has to prove he's the real one to Bryce by talking in Klingon which he's rusty at . At Ellie's Thanksgiving dinner , Anna shows jealousy at seeing Morgan's looks at Ellie as well as the way he's REALLY enjoying her food . Chuck also is envious watching secretly Sarah and Bryce ( who's unbeknownst to everyone else ) kissing in his bedroom , so much so he's tells Casey in encoded form about it . Chuck says " pineapple " to warn his fellow employees of an emergency as he's being held up by some enemy agents threatening to kill everyone in sight . Morgan awkwardly seeing Bryce and saying how he reminds him of Chuck's old roommate who he thinks is a " douche " . And then seeing Jeff hit on the head a couple of times after revealing how he liked being hit by a pineapple by his dad during his childhood . The episode ends with Sarah having to choose between two calls : One from Bryce on a classic antique phone made for mansions or on her ipod phone from Chuck . What to pick ? . . . Most hilarious and action-packed of " Chuck " yet ! With all my favorite characters and thousands of soccer mom shoppers contributing to the fun ! Very glad to know NBC gave this one a full-season pickup . Hope the writers ' strike ends before the episodes run out .
"""

# Bottle routes
@hook('after_request')
def enable_cors():
    response.headers['Access-Control-Allow-Origin'] = '*'

@route("/")
@view("index")
def index():
    return dict(example_text=example_text, anonymized=None, msg=None)

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
