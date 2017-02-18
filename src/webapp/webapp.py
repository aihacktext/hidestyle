#!/usr/bin/env python3
#
# Webapp
#

from datetime import datetime
from bottle import route, run, view, post, request

import classifier

clf = vectorizer = None


# Bottle routes

@route("/")
@view("index")
def index():
    return dict(orig_text='', anonymized=None, msg=None)

@post("/post")
@view("index")
def receive_form_post():
    orig_text = request.forms.get('text')
    anonymized, msg = classifier.classify_new_text(clf, vectorizer, orig_text)
    return dict(orig_text=orig_text, anonymized=anonymized, msg=msg)


def main():
    global clf, vectorizer
    clf, vectorizer = classifier.load_pickles()

    run(host='localhost', port=8080, debug=True)

if __name__ == "__main__":
    main()
