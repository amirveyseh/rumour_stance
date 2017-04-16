import time, tweet, sys, operator
from kerasRNN import kerasRNN


class LMRNN():
    def __init__(self, labels=[], rumourIds=[], tweetIds=[], maxlen=37, vocabulary_size=800, embedding_size=128):
        self.labels = labels;
        self.rumourIds = rumourIds;
        self.tweetIds = tweetIds;
        self.maxlen = maxlen;
        self.vocabulary_size = vocabulary_size;
        self.embedding_size = embedding_size;

        self.embeddings_s = {};
        self.embeddings_d = {};
        self.embeddings_q = {};
        self.embeddings_c = {};

    def train(self, rememberEmbeddings=False):
        print >> sys.stderr, 'LMRNN Training...'
        t1 = time.time();
        # tweetTexts = tweet.readTweetContents(map(int, self.ememes), map(int, self.infecting_vec));
        # self.model = kerasRNN(texts=tweetTexts, labels=self.node_vec, static=True);
        # x_train = self.model.preprocess_text(texts=tweetTexts, createIndex=True);
        # ls = [];
        # for s in x_train:
        #     ls.append(len(s));
        # print >> sys.stderr, 'max : ', np.max(ls);


        tweetTexts = tweet.readTweetContents(map(int, self.rumourIds), map(int, self.tweetIds));
        self.lm_s = kerasRNN(texts=tweetTexts, labels=self.labels, binary=True, targetClass=0, maxlen=self.maxlen,
                             vocabulary_size=self.vocabulary_size, embedding_size=self.embedding_size);
        self.lm_d = kerasRNN(texts=tweetTexts, labels=self.labels, binary=True, targetClass=1, maxlen=self.maxlen,
                             vocabulary_size=self.vocabulary_size, embedding_size=self.embedding_size);
        self.lm_q = kerasRNN(texts=tweetTexts, labels=self.labels, binary=True, targetClass=2, maxlen=self.maxlen,
                             vocabulary_size=self.vocabulary_size, embedding_size=self.embedding_size);
        self.lm_c = kerasRNN(texts=tweetTexts, labels=self.labels, binary=True, targetClass=3, maxlen=self.maxlen,
                             vocabulary_size=self.vocabulary_size, embedding_size=self.embedding_size);

        if rememberEmbeddings:
            self.predict_test(None,None,None,self.tweetIds,self.rumourIds,None,None,[],rememberEmbeddings=True);

        t2 = time.time();
        print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.);

    def evaluate_stance(
            self,
            testN,
            testtimes,
            testinfected_vec,
            testinfecting_vec,
            testeventmemes,
            testW,
            testT
    ):
        return self.predict_test(
            testN,
            testtimes,
            testinfected_vec,
            testinfecting_vec,
            testeventmemes,
            testW,
            testT)

    def predict_test(
            self,
            testN,
            testtimes,
            testinfected_vec,
            testinfecting_vec,
            testeventmemes,
            testW,
            testT,
            testnode_vec,
            rememberEmbeddings=False
    ):
        predictednode_vec = [None for _ in xrange(len(testinfecting_vec))]
        tweetTexts = tweet.readTweetContents(map(int, testeventmemes), map(int, testinfecting_vec))
        index = -1;
        for tt in tweetTexts:
            index += 1;

            s_score = self.lm_s.predict(tt, score=True);
            d_score = self.lm_d.predict(tt, score=True);
            q_score = self.lm_q.predict(tt, score=True);
            c_score = self.lm_c.predict(tt, score=True);

            scores = [s_score, d_score, q_score, c_score];
            max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))

            predictednode_vec[index] = max_index;

            if rememberEmbeddings:
                self.embeddings_s[testinfecting_vec[index]] = self.lm_s.getEmbeddings(tt);
                self.embeddings_d[testinfecting_vec[index]] = self.lm_d.getEmbeddings(tt);
                self.embeddings_q[testinfecting_vec[index]] = self.lm_q.getEmbeddings(tt);
                self.embeddings_c[testinfecting_vec[index]] = self.lm_c.getEmbeddings(tt);

        return predictednode_vec

        # return np.zeros((len(testtimes,)))+3;

    def getEmbedding(self, tweetId):

        if tweetId in self.embeddings_s.keys():
            embedding_s = self.embeddings_s[tweetId];
        else:
            raise ValueError('No embedding for the ', str(tweetId), ' tweet ID!')

        if tweetId in self.embeddings_d.keys():
            embedding_d = self.embeddings_d[tweetId];
        else:
            raise ValueError('No embedding for the ', str(tweetId), ' tweet ID!')

        if tweetId in self.embeddings_q.keys():
            embedding_q = self.embeddings_q[tweetId];
        else:
            raise ValueError('No embedding for the ', str(tweetId), ' tweet ID!')

        if tweetId in self.embeddings_c.keys():
            embedding_c = self.embeddings_c[tweetId];
        else:
            raise ValueError('No embedding for the ', str(tweetId), ' tweet ID!')

        return embedding_s, embedding_d, embedding_q, embedding_c;