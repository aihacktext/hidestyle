from nltk.corpus import wordnet as wn
from gensim.models import Doc2Vec, doc2vec, word2vec
import nltk

# I apologise for my naming conventions
from cleanse_text import normalize_text

# load the doc2vec model of the training set
d2v_model = Doc2Vec.load("d2v_model.doc2vec")

def load_lime_words():
    # stub
    lime_words = ['exciting', 'fantastic', 'mindset', 'movie', 'masterpiece']
    return nltk.pos_tag(lime_words)

lime_words = load_lime_words()

# load input data
input_doc = "To put it simply, the movie is fascinating, exciting and fantastic. The dialog, the fight choreography, the way the story moves, the characters charisma, all and much more are combined together to deliver this masterpiece. Such an amazing flow, providing a fusion between the 90s and the new century, it's like the assassins are living in another world, with another mindset, without people understanding it. Just one advice for you though: Don't build an expectation of what you want to watch in this movie, if you do, then you will ruin it. This movie has it's own flow and movement, so watch it with a clear mind, and have fun."


most_similar(nlp.vocab[u'dog'])

#This function is taken from github: https://github.com/explosion/spaCy/issues/276
def most_similar(word):
  by_similarity = sorted(word.vocab, key=lambda w: word.similarity(w), reverse=True)
  return [w.orth_ for w in by_similarity[:10]]

def syn_clean(word):
  nlp = spacy.load('en')
  return [s for s in list(set([w.lower() for w in most_similar(nlp.vocab[word])])) if s != word][1]


def replace_words(input_doc, lime_words, nlp):
    for word, tag in lime_words:
        best_word = syn_clean(word)        

    input_doc.replace(word, best_word)
    return input_doc




orig = replace_words(input_doc, lime_words, nlp)

