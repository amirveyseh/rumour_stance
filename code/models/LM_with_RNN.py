import itertools
import nltk
# from utils import *
import numpy as np
from rnn_theano import RNNTheano
import time
import sys

_VOCABULARY_SIZE = 800;
_HIDDEN_DIM = 80;
_LEARNING_RATE = 0.005;
_NEPOCH = 10
# _MODEL_FILE = "";

class LM_With_RNN:
    def __init__(self, texts):

        # Create the training data
        xy = self.preprocess_text(texts);
        self.X_train = xy['x'];
        self.y_train = xy['y'];

        self.model = RNNTheano(_VOCABULARY_SIZE, hidden_dim=_HIDDEN_DIM)

        self.train_with_sgd()

    def train_with_sgd(self, nepoch=_NEPOCH, evaluate_loss_after=5, learning_rate=_LEARNING_RATE):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.model.calculate_loss(self.X_train, self.y_train)
                losses.append((num_examples_seen, loss))
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
            # For each training example...
            for i in range(len(self.y_train)):
                # One SGD step
                self.model.sgd_step(self.X_train[i], self.y_train[i], learning_rate)
                num_examples_seen += 1
        return self.model;

    def calculate_score(self, text):
        texts = [text];
        xy = self.preprocess_text(texts);
        X_train = xy['x'];
        y_train = xy['y'];
        o = self.model.forward_propagation(X_train[0])
        p = 0;
        i = -1;
        for w in X_train[0]:
            i += 1;
            p += -1 * np.log10(o[i][w])
        return p;

    def preprocess_text(self, texts, vocabulary_size=_VOCABULARY_SIZE):
        unknown_token = "UNKNOWN_TOKEN"
        sentence_start_token = "SENTENCE_START"
        sentence_end_token = "SENTENCE_END"

        # Split full comments into sentences
        # sentences = itertools.chain(*[nltk.sent_tokenize(x.decode('utf-8').lower()) for x in texts])
        sentences = itertools.chain(*[nltk.sent_tokenize(x.lower()) for x in texts])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

        # Get the most common words and build index_to_word and word_to_index vectors
        if(vocabulary_size==-1):
            vocab = word_freq.elements();
        else:
            vocab = word_freq.most_common(vocabulary_size - 1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

        return {
            'x': X_train,
            'y': y_train,
            'index_to_word': index_to_word,
            'word_to_index': word_to_index
        };

