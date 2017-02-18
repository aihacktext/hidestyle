"""
Script to train an author verification model on a new review that identifies the new author versus all existing authors.

Uses Python 3.5
"""

import argparse
import sys
import os

import numpy as np
import scipy.sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import nltk
from lime.lime_text import LimeTextExplainer

sent_splitter = nltk.data.load('tokenizers/punkt/english.pickle')


# TODO replace named entities in the data and the target review

def main(args):

  parser = argparse.ArgumentParser(description='Train an author verification model.')
  parser.add_argument('-d', '--data-path', required=True, help='the path to the data directory')
  parser.add_argument('-r', '--review-type', default='imdb', help='the type of the review that shoudl be used',
                      choices=['imdb', 'pang'])
  parser.add_argument('-n', '--num-examples-per-class', default=10000, type=int,
                      help='number of training examples per class')
  parser.add_argument('-l', '--num-lime-features', default=10, type=int,
                      help='the top features used by LIME')
  args = parser.parse_args(args)

  assert os.path.exists(args.data_path), 'Error: %s does not exist.' % args.data_path
  if args.review_type == 'imdb':
    # this is the IMDb62 dataset containing of 62k reviews
    data_path = os.path.join(args.data_path, 'imdb62.txt')
    read_data = read_imdb_reviews
  else:
    # this is the dataset of movie reviews by Pang et al. (2003)
    data_path = os.path.join(args.data_path, 'movie-reviews-pang')
    read_data = read_movie_reviews_pang
  assert os.path.exists(data_path), 'Error: %s does not exist.' % data_path

  print('Reading movie reviews from %s...' % args.data_path)
  reviews = read_data(data_path)

  print('Extracting features from reviews...')
  vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), max_features=10000, stop_words='english')
  # Note: we use word vectorizer at the moment, but we could also use the below character ngram vectorizer
  # vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 5), max_features=50000)

  print('Using %d random reviews...' % args.num_examples_per_class)
  rand_indices = np.random.permutation(range(len(reviews)))[:args.num_examples_per_class]
  reviews = [review for (i, review) in enumerate(reviews) if i in rand_indices]
  rand_review_features = vectorizer.fit_transform(reviews)
  print('Review features shape:', rand_review_features.shape)

  while True:
    # Note: the final model workflow would take an API input, but here we just pre-define a new review for faster
    # prototyping
    # user_input = input('Please write a review:')
    user_input = """Believe it or not , I watched this show in the beginning in 1996 and I was a fan of it then . Not anymore , I grew to lose respect for all of the co-hosts at one time or another . Never did I think that Meredith Viera would announce to the world that she is not wearing underwear . Debbie Matenopoulos was far better than her replacements despite her sometimes idiotic comments . Until yesterday when Star Jones Reynolds announced her departure from the show did it hit me , I can't stand Rosie O'Donnell to begin with because she is a lying hypocrite of a human being and for her to succeed Viera's departure only supports my decision to have stopped watching . Who wants to see Rosie O'Donnell again as a talk show host anyway ? That was what was behind Star's departure was the Rosie's arrival . Not that Star is herself blameless , she has changed a lot since her surgery and her marriage but she managed to maintain some of my respect . After all , she is a lawyer and I don't think she's no dummy . Barbara Walters , what have you done to this show since it first aired 9 years ago . I could see why Meredith is leaving for better opportunities . I can't stand Joy Behar anymore who I used to enjoy watching as a comedian . I had respect for Star Jones until this show and watched it deplete over the years . Debbie's replacements have never had the same magic as the show once did when it premiered and even less when Rosie joins the show . With Star , I wish her and Meredith the best . But come on , Barbara , you can go outside and get somebody off the streets of New York City who doesn't talk English and do a better job then Rosie . They could be homeless , drunk , and wasted and I bet they would be funnier and more original than Rosie O'Donnell could ever be . I was reading that you were going to ask Marcia Cross about those lesbian rumors in front of her parents even though she was engaged to a man at the time but Marcia's quite a lady but you lost her respect . How many other people's respect are you going to lose now . Maybe Barbara should retire herself . Now if Barbara was smart , she would have gotten Kathie Lee Gifford to take over Meredith's spot . I plan on watching something less argumentative , maybe I'll switch over to Jerry Springer from now on ."""
    # user_input = "The movie is one of the greatest I ever saw. Sooo good.!!!"

    user_features = vectorizer.transform([user_input])
    # Note: We could split the user review into more sentences to get more data, but this does not seem to help.
    # user_features = vectorizer.transform(split_text_in_sentences(user_input))
    print('User features shape:', user_features.shape)
    author_verifier = LogisticRegression()

    # we duplicate the user review to generate a balanced training data set
    duplicated_user_features = scipy.sparse.vstack([user_features for _ in range(int(args.num_examples_per_class / user_features.shape[0]))])
    print('Shape of duplicated user features:', duplicated_user_features.shape)

    train_data = scipy.sparse.vstack([rand_review_features, duplicated_user_features])
    print('Train data shape:', train_data.shape)

    # Note: We use 0 for the generic author and 1 for the new author.
    class_names = ['Generic-author', 'New-author']
    train_labels = np.array([0] * args.num_examples_per_class + [1] * duplicated_user_features.shape[0])
    print('Train labels shape:', train_labels.shape)

    print('Training author verification model on data...')
    author_verifier.fit(train_data, train_labels)

    # we create a pipeline for prediction and for the Lime Explainer
    c = make_pipeline(vectorizer, author_verifier)

    print('Prediction on target document:', c.predict_proba([user_input])[0])
    explainer = LimeTextExplainer(class_names=class_names)

    # we use LIME to identify the features that are most indicative for both classes
    exp = explainer.explain_instance(user_input, c.predict_proba, num_features=args.num_lime_features)
    print(exp.as_list())

    new_user_input = user_input
    for (word, prob) in exp.as_list():
      # we can now replace all features that have a positive probability (are indicative of teh New-author class)
      # and observe how the probability of predicting the New-author changes
      if prob > 0:
        print('Replacing word %s...' % word)
        new_user_input = new_user_input.replace(word, ' ')
    print(c.predict_proba([new_user_input])[0])
    sys.exit(0)


def read_movie_reviews_pang(data_path):
  """
  Read reviews from the Pang et al. corpus.
  :param data_path: the path to the movie-reviews-pang directory
  :return: a list of reviews
  """
  pos_file = os.path.join(data_path, 'rt-polarity.pos')
  neg_file = os.path.join(data_path, 'rt-polarity.neg')
  reviews = []
  for file_path in [pos_file, neg_file]:
    with open(file_path, encoding='latin-1') as f:
      for line in f:
        reviews.append(line.strip())
  return reviews


def read_imdb_reviews(data_path):
  """
  Read reviews from the IMBb62 corpus file.
  :param data_path: the path to the imdb62.txt file
  :return: a list of reviews
  """
  reviews = []
  with open(data_path, encoding='utf-8') as f:
    for line in f:
      review = line.split('\t')[5].strip()
      reviews.append(review)
  return reviews


def split_text_in_sentences(text):
    """
    Splits a text in sentences. Merges sentences if a sentence only consists of one token.
    :param text: the input text
    :return: a list of sentences
    """
    sentences = sent_splitter.tokenize(text)
    t_sentences = []
    for i in range(len(sentences)):
        if len(nltk.word_tokenize(sentences[i])) > 1 or i == 0:
            t_sentences.append(sentences[i])
        else:
            t_sentences[-1] += ' ' + sentences[i]
    return t_sentences


if __name__ == '__main__':

  main(sys.argv[1:])