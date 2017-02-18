# spacy lightning tour

# lemmas, part of speech tagging, ranking of words etc
import spacy

# maybe
from nltk.corpus import wordnet as wn
# just for dev/testing
from nltk.corpus import brown
# maybe
from nltk.tokenize import casual_tokenize
# sentence tokenizer, general useful nlp stuff
import nltk

# for word2vec, doc2vec
from gensim.models import doc2vec

# general handy python stuff
import re
import logging
import pandas as pd
import imp

# quick and dirty analysis of writing styles
import stylometry as st

# generate new sentences. Might use this to blend writing styles together
import markovify

# markovify is a work in progress. import from the source code
imp.reload(markovify)
from markovify.text import POSifiedText, SpacyPOSifiedText
from stylometry.extract import StyloString

#-------------------------------------------------------------------------------------------------#
# stuff I might use

# load english tokeniser, tagger, parser, NER, word vectors
# nlp = spacy.load('en')
# spacy_model= nlp(text)
# [(a.text, a.lemma, a.lemma_, a.tag_, a.pos_) for a in spacy_model]

# different words for stuff
dog = wn.synset('dog.n.01')

#-------------------------------------------------------------------------------------------------#
# gensim and word2vec

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

# read in a huge collection of text
text = open("texts/war_and_peace.txt", "r").read().lower()
text = normalize_text(text)

# split into sentences
nltk_sentences = nltk.sent_tokenize(text)

nltk_sentences = [doc2vec.TaggedDocument(words.split(), [i]) \
                for i, words in enumerate(nltk_sentences)]


# Doc2Vec model initialisation
d2v_model = doc2vec.Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5,
                    hs=0, min_count=10, workers=4)

# "build" the model with our chosen set of text
d2v_model.build_vocab(nltk_sentences)

# Very simple usage of doc2vec: Look for words similar to this one
d2v_model.most_similar(['her'])

"""
TODO:
for word in sentence:
    if word.lemma > some_cutoff and word.pos_ != proper_noun:
        word.text = model.most_similar(word.text, topn=1)
"""

def analyze_string(sentence):
    """
    List of tuples of some of the stylometry analysis output
    """
    style = st.extract.StyloString(sentence)
    out = zip(style.csv_header().split(','), style.csv_output().split(','))
    res = [a for a in out]

    # relevant stuff
    idx =[2, 3, 4, 5, 6, 7]
    return [res[i] for i in idx]



