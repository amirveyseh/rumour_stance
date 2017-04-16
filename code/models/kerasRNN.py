'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
# from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import itertools
import nltk
from nltk.corpus import stopwords
import numpy as np
import sys;
import keras.backend as K

# _max_features = 20000
_maxlen = 37  # cut texts after this number of words (among top max_features most common words)
_batch_size = 100
_VOCABULARY_SIZE = 800;
_unknown_token = "UNKNOWN_TOKEN"
_number_epoch = 3;


class kerasRNN:
    def __init__(self, texts=[], labels=[], n_epoch=_number_epoch, batch_size=_batch_size, vocabulary_size=_VOCABULARY_SIZE, maxlen=_maxlen, embedding_size=128, static=False, binary=False, targetClass=0):
        if(static is not True):
            np.random.seed(0);

            self.maxlen = maxlen;

            x_train = self.preprocess_text(texts, vocabulary_size=vocabulary_size, createIndex=True);
            if(binary):
                outputDim = 1;
                lossFunc = 'binary_crossentropy';
                activFunc = 'sigmoid';
                y_train = labels == targetClass;
                y_train = np.asarray(y_train).astype(int);
            else:
                outputDim = 4;
                lossFunc = 'categorical_crossentropy';
                activFunc = 'softmax';
                y_train = np.zeros((len(labels), 4), dtype=int);
                for i in xrange(len(labels)):
                    y_train[i][labels[i]] = 1;
            assert len(x_train) == len(y_train)
            # print('Loading data...')
            # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
            # print >> sys.stderr, (x_train, y_train);
            print(len(x_train), 'train sequences')
            # print(len(x_test), 'test sequences')

            print('Pad sequences (samples x time)')
            x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
            # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
            print('x_train shape:', x_train.shape)
            # print('x_test shape:', x_test.shape)

            print('Build model...')
            model = Sequential()
            model.add(Embedding(vocabulary_size, embedding_size, input_length=self.maxlen))
            model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(outputDim, activation=activFunc))

            # try using different optimizers and different optimizer configs
            model.compile(loss=lossFunc,
                          optimizer='RMSprop',
                          metrics=['accuracy'])

            print('Train...')
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=n_epoch,
                      validation_split=0.2)
            # score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
            # print('Test score:', score)
            # print('Test accuracy:', acc)

            self.model = model;

    def predict(self, text, score=False):
        texts = [text];
        x_test = self.preprocess_text(texts);
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        if(score):
            probs = self.model.predict_proba(x_test)[0][0];
            return probs;
        else:
            label = self.model.predict_classes(x_test)[0];
            return label;

    def getEmbeddings(self, text):
        texts = [text];
        x_test = self.preprocess_text(texts);
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        get_Embedding_output = K.function([self.model.layers[0].input],
                                          [self.model.layers[0].output]);
        return get_Embedding_output([x_test])[0][0]

    def preprocess_text(self, texts, vocabulary_size=_VOCABULARY_SIZE, createIndex=False):
        tokenized_sentences = self.tokenize(texts);

        if (createIndex):
            self.word_to_index = self.createIndex(tokenized_sentences, vocabulary_size=vocabulary_size);

        word_to_index = self.word_to_index;

        tokenized_sentences = self.normalizeText(tokenized_sentences, word_to_index);

        # Create the training data
        X_train = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])

        return X_train;

    def tokenize(self, texts):
        # sentence_start_token = "SENTENCE_START"
        # sentence_end_token = "SENTENCE_END"

        # Split full comments into sentences
        # sentences = itertools.chain(*[nltk.sent_tokenize(x.decode('utf-8').lower()) for x in texts])
        # print >> sys.stderr, 'len texts: ', len(texts);
        # sentences = itertools.chain(*[nltk.sent_tokenize(x.lower()) for x in texts])
        # count = 0;
        # for _ in sentences:
        #     count += 1;
        # print >> sys.stderr, 'len sentences: ', count;
        # Append SENTENCE_START and SENTENCE_END
        # sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
        # print >> sys.stderr, 'len sentences: ', len(sentences);

        # Tokenize the sentences into words
        # tokenized_sentences = [nltk.word_tokenize(text) for text in texts]
        tokenized_sentences = [nltk.wordpunct_tokenize(text.lower()) for text in texts]
        newTokens = [];
        for sent in tokenized_sentences:
            newSent = [];
            for token in sent:
                newToken = [];
                splits = token.split("-");
                for spl in splits:
                    spls = spl.split("_");
                    newToken.extend(spls);
                newSent.extend(newToken);
            newTokens.append(newSent);
        tokenized_sentences = newTokens;
        for i, x in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [word for word in x if (word not in stopwords.words('english')) or (word in ['no', 'not']) ]

        return tokenized_sentences;

    def normalizeText(self, tokenized_sentences, word_to_index):
        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else _unknown_token for w in sent]

        return tokenized_sentences;

    def createIndex(self, tokenized_sentences, vocabulary_size=_VOCABULARY_SIZE):
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

        # print >> sys.stderr, word_freq.r_Nr();
        # print >> sys.stderr, word_freq.B();

        # Get the most common words and build index_to_word and word_to_index vectors
        if (vocabulary_size == -1):
            vocab = word_freq.keys();
            index_to_word = [x for x in vocab]
        else:
            vocab = word_freq.most_common(vocabulary_size - 1)
            index_to_word = [x[0] for x in vocab]
        index_to_word.append(_unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        return word_to_index;









































# texts = ['this is amir.', 'is it reza?', 'she is rahele.', 'they are family.', 'this is baba.']
# labels = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
# test = ['this is omid.'];
# kr = kerasRNN(static=True);
# x_train = kr.preprocess_text(texts, createIndex=True, vocabulary_size=10);
# x_test = kr.preprocess_text(test, createIndex=True, vocabulary_size=10);
# print x_train.shape;
#
#
# model = Sequential()
# model.add(Embedding(10, 3, input_length=4))
# model.add(LSTM(4, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(1, activation='softmax'))
#
# # model.compile('rmsprop', 'mse')
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print('Train...')
# model.fit(x_train, labels,
#           epochs=10)
# output_array = model.predict(x_train)
# print output_array;


# '''Trains a LSTM on the IMDB sentiment classification task.
# The dataset is actually too small for LSTM to be of any advantage
# compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes:
# - RNNs are tricky. Choice of batch size is important,
# choice of loss and optimizer is critical, etc.
# Some configurations won't converge.
# - LSTM loss decrease patterns during training can be quite different
# from what you see with CNNs/MLPs/etc.
# '''
# # from __future__ import print_function
# #
# # from keras.preprocessing import sequence
# # from keras.models import Sequential
# # from keras.layers import Dense, Embedding
# # from keras.layers import LSTM
# # from keras.datasets import imdb
#
# max_features = 20000
# maxlen = 80  # cut texts after this number of words (among top max_features most common words)
# batch_size = 32
#
# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, )
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
#
# print "Preprocess...."
# x_train = x_train[:10,:];
# # y_train = y_train[:10];
# y_train = np.zeros((10,4),dtype=int);
# for i in xrange(10):
#     j = i%4;
#     y_train[i][j] = 1;
# x_test = x_test[:5,:];
# # y_test = y_test[:5];
# y_test = np.zeros((5,4),dtype=int);
# for i in xrange(5):
#     j = i%4;
#     y_test[i][j] = 1;
# x_train = x_train[:,:,np.newaxis]
# x_test = x_test[:,:,np.newaxis]
#
#
#
# print 'debug :::::::::::::::::: ';
# print 'x_train.shape', x_train.shape;
# print 'y_train.shape', y_train.shape;
# # print '************* 0 *************'
# # print 'x_train[0]', x_train[0];
# # print 'y_train[0]', y_train[0];
# # print '************* 1 *************'
# # print 'x_train[1]', x_train[1];
# # print 'y_train[1]', y_train[1];
# # print '************* 2 *************'
# # print 'x_train[2]', x_train[2];
# # print 'y_train[2]', y_train[2];
# # print '************* 3 *************'
# # print 'x_train[3]', x_train[3];
# # print 'y_train[3]', y_train[3];
# # print '************* 4 *************'
# # print 'x_train[4]', x_train[4];
# # print 'y_train[4]', y_train[4];
# # print '************* 5 *************'
# # print 'x_train[5]', x_train[5];
# # print 'y_train[5]', y_train[5];
# print 'debug :::::::::::::::::: ';
#
# print('Build model...')
# model = Sequential()
# # model.add(Embedding(max_features, 128))
# model.add(LSTM(2, dropout=0.2, recurrent_dropout=0.2, input_shape=(80, 1)))
# model.add(Dense(4, activation='softmax'))
#
# # try using different optimizers and different optimizer configs
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# print('Train...')
# model.fit(x_train, y_train,
#           batch_size=10,
#           epochs=1)
# # score, acc = model.evaluate(x_test, y_test,
# #                             batch_size=batch_size)
# # print('Test score:', score)
# # print('Test accuracy:', acc)
#
# print "Predict...."
# output_array = model.predict_classes(x_test)
# print output_array.shape;
# print output_array;




from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim), recurrent_dropout=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(10, activation='softmax'))  # return a single vector of dimension 32
# model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64)