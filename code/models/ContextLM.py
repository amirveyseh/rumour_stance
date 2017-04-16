from seq2seqRNN import seq2seqRNN
from LMRNN import LMRNN
import numpy as np, sys, time, operator
from keras.models import Sequential
from keras.layers import LSTM, Dense

_BINARY = True;
_BATCH_SIZE = 32;
_EPOCH = 3;
_MAXLEN = 37;
_VOCABULARY_SIZE = 800;
_EMPEDDING_SIZE = 128;
_HIDDEN_HISTORY_SIZE = 128;


class ContextLM():
    def __init__(self, rumourIds=[], tweetIds=[], infected_vec=[], labels=[], times=[]):
        self.rumourIds = rumourIds;
        self.tweetIds = tweetIds;
        self.infected_vec = infected_vec;
        # self.labels = map(int, labels);
        self.labels = labels;
        self.times = times;

        self.seq2seqModel = seq2seqRNN(times=self.times, labels=self.labels, rumourIds=self.rumourIds,
                                       tweetIds=self.tweetIds, infected_vec=self.infected_vec,
                                       hidden_history_size=_HIDDEN_HISTORY_SIZE);
        self.LMModel = LMRNN(labels=self.labels, rumourIds=self.rumourIds, tweetIds=self.tweetIds,
                             vocabulary_size=_VOCABULARY_SIZE, maxlen=_MAXLEN, embedding_size=_EMPEDDING_SIZE)

        self.binary = _BINARY;
        self.vocabulary_size = _VOCABULARY_SIZE;
        self.maxlen = _MAXLEN;
        self.embedding_size = _EMPEDDING_SIZE;
        self.hidden_history = _HIDDEN_HISTORY_SIZE;

    def train(self):
        print >> sys.stderr, 'ContextLM Training...'
        t1 = time.time()

        self.seq2seqModel.train(rememberHistory=True);
        self.LMModel.train(rememberEmbeddings=True);

        if self.binary:
            x_train_s, x_train_d, x_train_q, x_train_c, y_train_s, y_train_d, y_train_q, y_train_c = self.extractFeaturesLabels(self.seq2seqModel,
                                                                                             self.LMModel,
                                                                                             self.tweetIds);
        else:
            x_train, y_train = self.extractFeaturesLabels(self.seq2seqModel, self.LMModel, self.tweetIds);

        if self.binary:
            self.model_s = self.trainModel(x_train_s, y_train_s);
            self.model_d = self.trainModel(x_train_d, y_train_d);
            self.model_q = self.trainModel(x_train_q, y_train_q);
            self.model_c = self.trainModel(x_train_c, y_train_c);
        else:
            self.model = self.trainModel(x_train, y_train);

        t2 = time.time()
        print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.)

    def trainModel(self, x_train, y_train, batch_size=_BATCH_SIZE, n_epoch=_EPOCH):

        print >> sys.stderr, np.asarray(x_train).shape
        print >> sys.stderr, np.asarray(y_train).shape
        print >> sys.stderr, y_train
        print >> sys.stderr, self.labels
        print >> sys.stderr, (self.maxlen, self.embedding_size + self.hidden_history)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        if self.binary:
            outDim = 1;
            activationFunc = 'sigmoid';
            lossFunc = 'binary_crossentropy'
        else:
            outDim = 4;
            activationFunc = 'softmax';
            lossFunc = 'categorical_crossentropy'

        model = Sequential()
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,
                       batch_input_shape=(None, self.maxlen, self.embedding_size + self.hidden_history)))
        model.add(Dense(outDim, activation=activationFunc))

        model.compile(loss=lossFunc, optimizer='RMSprop', metrics=['accuracy'])

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=n_epoch,
                  validation_split=0.2)

        return model;

    def extractFeaturesLabels(self, seq2seqModel, LMModel, tweetIds, makeTarget=True):
        if self.binary:
            x_train_s = []
            x_train_d = []
            x_train_q = []
            x_train_c = []
        else:
            x_train = [];

        for ind, tweetId in enumerate(tweetIds):
            historyState = seq2seqModel.getHistoryState(tweetId);
            if self.binary:
                embeddings_s, embeddings_d, embeddings_q, embeddings_c = LMModel.getEmbedding(tweetId);
            else:
                embeddings = LMModel.getEmbedding(tweetId);
            if self.binary:
                x_train_s.append(map(lambda embedding: list(np.concatenate((embedding, historyState), axis=0)), embeddings_s));
                x_train_d.append(map(lambda embedding: list(np.concatenate((embedding, historyState), axis=0)), embeddings_d));
                x_train_q.append(map(lambda embedding: list(np.concatenate((embedding, historyState), axis=0)), embeddings_q));
                x_train_c.append(map(lambda embedding: list(np.concatenate((embedding, historyState), axis=0)), embeddings_c));
            else :
                x_train.append(map(lambda embedding: list(np.concatenate((embedding, historyState), axis=0)), embeddings));


        if makeTarget:
            if self.binary:
                y_train_s = self.labels == 0;
                y_train_s = list(np.asarray(y_train_s).astype(int));
                y_train_d = self.labels == 1;
                y_train_d = list(np.asarray(y_train_d).astype(int));
                y_train_q = self.labels == 2;
                y_train_q = list(np.asarray(y_train_q).astype(int));
                y_train_c = self.labels == 3;
                y_train_c = list(np.asarray(y_train_c).astype(int));
                return x_train_s, x_train_d, x_train_q, x_train_c, y_train_s, y_train_d, y_train_q, y_train_c;
            else:
                y_train = map(lambda label: [1 if i == label else 0 for i in xrange(4)], self.labels);
                return x_train, y_train;
        else:
            if self.binary:
                return x_train_s, x_train_d, x_train_q, x_train_c;
            else:
                return x_train;

    def evaluate_stance(
            self,
            testN,
            testtimes,
            testinfected_vec,
            testinfecting_vec,
            testeventmemes,
            testW,
            testT,
            testnode_vec
    ):

        predictednode_vec = np.zeros((len(testtimes),), dtype=int);

        self.seq2seqModel.predict_test(
            testN,
            testtimes,
            testinfected_vec,
            testinfecting_vec,
            testeventmemes,
            testW,
            testT,
            testnode_vec,
            rememberHistory=True);

        self.LMModel.predict_test(
            testN,
            testtimes,
            testinfected_vec,
            testinfecting_vec,
            testeventmemes,
            testW,
            testT,
            testnode_vec,
            rememberEmbeddings=True);

        x_test_s, x_test_d, x_test_q, x_test_c = self.extractFeaturesLabels(self.seq2seqModel, self.LMModel, testinfecting_vec, makeTarget=False);
        x_test_s = np.asarray(x_test_s);
        x_test_d = np.asarray(x_test_d);
        x_test_q = np.asarray(x_test_q);
        x_test_c = np.asarray(x_test_c);

        # print >> sys.stderr, x_test.shape

        for ind in xrange(x_test_s.shape[0]):
            # print >> sys.stderr, np.asarray(test).shape
            if self.binary:
                s_score = self.model_s.predict_proba(np.asarray([x_test_s[ind]]))[0][0];
                d_score = self.model_d.predict_proba(np.asarray([x_test_d[ind]]))[0][0];
                q_score = self.model_q.predict_proba(np.asarray([x_test_q[ind]]))[0][0];
                c_score = self.model_c.predict_proba(np.asarray([x_test_c[ind]]))[0][0];
                scores = [s_score, d_score, q_score, c_score];
                max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))
                predictednode_vec[ind] = max_index;
            # else:
            #     predictednode_vec[ind] = self.model.predict_classes(np.asarray([test]))[0];

        print >> sys.stderr, predictednode_vec;
        print >> sys.stderr, testnode_vec;

        return predictednode_vec
