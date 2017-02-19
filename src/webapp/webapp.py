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
from text_processing.conservative_word_replace import *
from classification import classifier
from text_processing.style import analyze_string

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

def anonymize(orig_text):
    mangled = normalize_text(orig_text)
    anonymized, msg = classifier.classify_new_text(clf, vectorizer, mangled)
    anonymized = ReplaceSomeWords(anonymized)
    hot_words = classifier.lime_explain(clf, vectorizer, mangled, num_examples_per_class=20,
                 num_lime_features=20)

    style = analyze_string(orig_text)
    if hot_words:
        msg = "Better choose some synonyms for: %s" % ' '.join(hot_words)
    else:
        msg = "Performing conservative phrase replacement"
    return dict(orig_text=orig_text, anonymized=anonymized, msg=msg, style=style)

@route("/anonymize")
def serve_anonymize():
    orig_text = request.query.text
    return anonymize(orig_text)

@route('/scripts/<filename>')
def server_static(filename):
    return static_file(filename, root='scripts/')

def learn(max_items_per_author):
    global clf, vectorizer
    clf, vectorizer = classifier.learn(
        max_items_per_author,
    )
    clf_fname, vectorizer_fname = classifier.generate_pkl_filenames()
    classifier.save_pickles(clf, clf_fname, vectorizer, vectorizer_fname)

@post("/learn")
def serve_learn():
    max_items_per_author = int(request.forms.get('max_items_per_author'))
    learn(max_items_per_author)
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
        learn(40)
        # Run anonymize once as a test
        print(anonymize(example_text))

    run(host='localhost', port=8080, debug=True, reloader=True)

if __name__ == "__main__":
    main()
