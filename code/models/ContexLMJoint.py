import tweet, rumours, numpy as np, sys, time
from gensim.models.word2vec import Word2Vec;
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.layers.wrappers import TimeDistributed;
from keras.layers.core import Masking;

_VOCABULARY_SIZE = 800;
_MAX_TWEET_LENGTH = 37;
_MAX_TWEET_SEQ_LENGTH = 103;
_EMBEDDING_SIZE = 128;
# _EMBEDDING_EPOCH = 1;
_MASKING_VALUE = 0
_BATCH_SIZE = 32;
_EPOCH = 1;

class ContextLMJoint():
    def __init__(self, rumourIds=[], tweetIds=[], infected_vec=[], labels=[], times=[], topic=-1):
        self.rumourIds = rumourIds;
        self.tweetIds = tweetIds;
        self.infected_vec = infected_vec;
        self.labels = labels;
        self.times = times;
        self.topic = topic;

    def train(self):
        print >> sys.stderr, 'ContextLM_Joint Training ...'
        t1 = time.time()
        x_train, y_train = self.prepare_data(self.rumourIds, self.tweetIds, self.times, labels=self.labels, make_target=True);
        print >> sys.stderr, "Training model ..."
        model = self.train_model(x_train, y_train);
        self.model = model;
        t2 = time.time()
        print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.)

    def train_model(self, x_train, y_train):
        sample_weights = self.compute_sample_weights(x_train);
        model = Sequential();
        model.add(TimeDistributed(Embedding(_VOCABULARY_SIZE+2,_EMBEDDING_SIZE,mask_zero=True,input_length=_MAX_TWEET_LENGTH), input_shape=(_MAX_TWEET_SEQ_LENGTH, _MAX_TWEET_LENGTH)))
        # model.add(TimeDistributed(Masking(mask_value=_MASKING_VALUE, input_shape=(_MAX_TWEET_LENGTH, _EMBEDDING_SIZE)), input_shape=(_MAX_TWEET_SEQ_LENGTH, _MAX_TWEET_LENGTH, _EMBEDDING_SIZE)))
        model.add(TimeDistributed(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(_MAX_TWEET_LENGTH, _EMBEDDING_SIZE))))
        model.add(Dense(10, activation='softmax'))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'], sample_weight_mode='temporal')
        model.fit(x_train, y_train, batch_size=_BATCH_SIZE, epochs=_EPOCH, validation_split=0.2, sample_weight=sample_weights)
        return model;

    def prepare_data(self, rumourIds, tweetIds, times, labels=[], make_target=True):
        maxtweetseq = 0;
        maxtweetlen = 0;
        # tweet_embeddings = self.extract_tweet_embeddings(rumourIds, tweetIds);
        tweets = tweet.preprocessTexts(tweet.readTweetContents(map(int, rumourIds), map(int, tweetIds), self.topic), vocabSize=_VOCABULARY_SIZE, remove_stop_words=True, mask_zero=True);
        rumour_seqs = rumours.extract_rumour_seqs(rumourIds, tweetIds, times);
        # x_train = np.zeros((len(set(rumourIds)), _MAX_TWEET_SEQ_LENGTH, _MAX_TWEET_LENGTH, _EMBEDDING_SIZE), dtype=float); # for _MASKING_VALUE 0.
        x_train = np.zeros((len(set(rumourIds)), _MAX_TWEET_SEQ_LENGTH, _MAX_TWEET_LENGTH), dtype=int); # for _MASKING_VALUE 0
        if make_target:
            y_train = np.zeros((len(set(rumourIds)), _MAX_TWEET_SEQ_LENGTH, 4), dtype=int);
        for rumourId, seq in enumerate(rumour_seqs):
            tweet_counter = 0;
            if len(seq) > maxtweetseq:
                maxtweetseq = len(seq);
            for tweet_ind, tweetId in enumerate(seq):
                # x_train[rumourId, tweet_ind,:,:] = np.asarray(tweet_embeddings[list(tweetIds).index(tweetId)])[0:_MAX_TWEET_LENGTH,:]
                tweet_tokens = tweets[list(tweetIds).index(tweetId)]
                if len(tweet_tokens) > maxtweetlen:
                    maxtweetlen = len(tweet_tokens);
                x_train[rumourId,tweet_ind,:len(tweet_tokens)] = tweet_tokens
                if make_target:
                    y_train[rumourId, tweet_ind, int(labels[list(tweetIds).index(tweetId)])] = 1;
                tweet_counter += 1;
                if tweet_counter >= _MAX_TWEET_SEQ_LENGTH:
                    break;
        # print >> sys.stderr, 'max tweet seq : ', maxtweetseq;
        # print >> sys.stderr, 'max tweet length : ', maxtweetlen;
        # print >> sys.stderr, x_train[0,0,:]
        # print >> sys.stderr, x_train[0,1,:]
        # print >> sys.stderr, x_train[1,2,:]
        if make_target:
            return x_train, y_train;
        else:
            return x_train;

    # def extract_tweet_embeddings(self, rumourIds, tweetIds):
    #     tweet_texts = tweet.readTweetContents(map(int, rumourIds), map(int, tweetIds), self.topic);
    #     tweet_texts = tweet.tokenize(tweet_texts);
    #     print >> sys.stderr, tweet_texts;
    #     assert True == False
    #     # vectors = Word2Vec(tweet_texts,size=_EMBEDDING_SIZE,max_vocab_size=_VOCABULARY_SIZE,iter=_EMBEDDING_EPOCH).wv;
    #     vectors = Word2Vec(tweet_texts,size=_EMBEDDING_SIZE,iter=_EMBEDDING_EPOCH).wv;
    #     embeddings = [];
    #     for tweet_text in tweet_texts:
    #         try:
    #             embeddings.append(vectors[tweet_text])
    #         except:
    #             print >> sys.stderr, tweet_text
    #             assert True==False
    #     return embeddings;

    def compute_sample_weights(self, x_train):
        x_train = np.asarray(x_train);
        sample_weights = np.ones((x_train.shape[0], x_train.shape[1]), dtype=int);
        for s_ind, sample in enumerate(x_train):
            for ts_ind, time_step in enumerate(sample):
                if np.all(time_step==_MASKING_VALUE):
                    sample_weights[s_ind,ts_ind] = 0;
        return sample_weights;

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
        x_test = self.prepare_data(testeventmemes, testinfecting_vec, testtimes, labels=[], make_target=False);
        predictednode_vec = self.model.predict_classes(x_test)[0,:len(testtimes)]
        print >> sys.stderr, predictednode_vec;
        print >> sys.stderr, testnode_vec
        return predictednode_vec;

        # return [3 for _ in testtimes]



