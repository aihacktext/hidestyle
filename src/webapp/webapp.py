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
    mangled = normalize_text(orig_text)
    anonymized, msg = classifier.classify_new_text(clf, vectorizer, orig_text)
    return dict(orig_text=orig_text, anonymized=mangled, msg=msg)

@route("/anonymize")
def anonymize():
    text = request.query.text
    anonymized = text
    msg = "Better choose some synonims for: foo bar baz"
    return dict(orig_text=text, anonymized=anonymized, msg=msg)

@route('/scripts/<filename>')
def server_static(filename):
    return static_file(filename, root='scripts/')

def main():
    global clf, vectorizer
    clf, vectorizer = classifier.load_pickles()

    run(host='localhost', port=8080, debug=True)

if __name__ == "__main__":
    main()
