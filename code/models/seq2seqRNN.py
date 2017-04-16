# import seq2seq
# from seq2seq.models import SimpleSeq2Seq, Seq2Seq
import sys, time, tweet, operator, nltk, numpy as np, itertools
from nltk.corpus import stopwords
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import LSTM, Dense
from Tree import Tree;
from keras import backend as K
from itertools import compress
from doc2vec.doc2vec import Doc2Vec

_VOCABULARY_SIZE = 800;
_UNKNOWN_TOKEN = "UNKNOWN_TOKEN";
# _BATCH_SIZE = 32;
_NEPOCH = 10;
_HIDDEN_HISTORY_SIZE=128;

# _MAX_LENGTH = 100;


class seq2seqRNN():
    def __init__(self, vocabulary_size=_VOCABULARY_SIZE, times=[], labels=[], rumourIds=[], tweetIds=[],
                 infected_vec=[], hidden_history_size=_HIDDEN_HISTORY_SIZE):
        self.etimes = times;
        self.node_vec = labels;
        self.ememes = rumourIds;
        self.infecting_vec = tweetIds;
        self.infected_vec = infected_vec;
        self.vocabulary_size = vocabulary_size;
        self.hidden_history_size = hidden_history_size;
        self.histories = {};
        # self.resample();
        self.doc2vec = Doc2Vec();

    def train(self, rememberHistory=False):
        print >> sys.stderr, 'seq2seqRNN Training...'
        t1 = time.time()

        self.binary = False;
        self.useTree = False;

        tweetTexts = tweet.readTweetContents(map(int, self.ememes), map(int, self.infecting_vec));
        # features = self.convertTextToFeature(tweetTexts);
        features = self.doc2vec.extract_train_vec(tweetTexts);
        self.input_shape = (1, 1, len(features[0]))
        if self.useTree:
            x_train, y_train = self.extract_threads_targets(features, self.infecting_vec, self.infected_vec,
                                                            self.node_vec, makeTarget=True);
        else:
            if self.binary:
                x_train, y_train_s, y_train_d, y_train_q, y_train_c = self.extract_seq_targets(features, self.etimes,
                                                                                               self.ememes,
                                                                                               self.node_vec,
                                                                                               makeTarget=True,
                                                                                               binary=True);
            else:
                x_train, y_train = self.extract_seq_targets(features, self.etimes, self.ememes, self.node_vec,
                                                            makeTarget=True);

        if self.binary:
            self.model_s = self.trainModel(x_train, y_train_s, binary=True);
            self.model_d = self.trainModel(x_train, y_train_d, binary=True);
            self.model_q = self.trainModel(x_train, y_train_q, binary=True);
            self.model_c = self.trainModel(x_train, y_train_c, binary=True);
        else:
            self.model = self.trainModel(x_train, y_train, binary=False);


        if rememberHistory:
            self.predict_test(None,self.etimes,self.infected_vec,self.infecting_vec,self.ememes,None,None,[],rememberHistory=True);

        t2 = time.time()
        print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.)

    def trainModel(self, x_train, y_train, binary):

        # model = Seq2Seq(input_shape=(1,1,self.vocabulary_size), hidden_dim=10, output_dim=4, output_length=_MAX_LENGTH, stateful=True)
        # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # model.fit(x_train, y_train, batch_size=1, shuffle=False, nb_epoch=_NEPOCH, validation_split=.2);

        if binary:
            outDim = 1;
            activationFunc = 'sigmoid';
            lossFunc = 'binary_crossentropy'
        else:
            outDim = 4;
            activationFunc = 'softmax';
            lossFunc = 'categorical_crossentropy'

        model = Sequential()
        model.add(LSTM(self.hidden_history_size, dropout=0.2, recurrent_dropout=0.2, stateful=True,
                       batch_input_shape=self.input_shape))
        model.add(Dense(outDim, activation=activationFunc))

        model.compile(loss=lossFunc, optimizer='RMSprop', metrics=['accuracy'])

        print('Train...')
        for epoch in range(_NEPOCH):
            # print >> sys.stderr, 'Epoch : ', epoch;
            for i in range(len(x_train)):
                for j in range(len(x_train[i])):
                    model.train_on_batch(np.asarray([[x_train[i][j]]]), np.asarray([y_train[i][j]]))
                model.reset_states()

        return model;

    def convertTextToFeature(self, texts):
        processedTexts = self.preprocessTexts(texts);
        features = self.extractFeatures(processedTexts);
        return features;

    def extractFeatures(self, texts):
        features = [];
        for text in texts:
            feature = np.zeros((self.vocabulary_size), dtype=int);
            for token in text:
                feature[token] = 1;
            features.append(feature);
        return features;

    def preprocessTexts(self, texts):
        tokenized_sentences = self.tokenize(texts);
        word_to_index = self.createIndex(tokenized_sentences, vocabulary_size=self.vocabulary_size);
        tokenized_sentences = self.normalizeText(tokenized_sentences, word_to_index);
        processedTexts = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
        return processedTexts;

    def tokenize(self, texts):
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
            tokenized_sentences[i] = [word for word in x if
                                      (word not in stopwords.words('english')) or (word in ['no', 'not'])]

        return tokenized_sentences;

    def normalizeText(self, tokenized_sentences, word_to_index):
        for i, sent in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else _UNKNOWN_TOKEN for w in sent]

        return tokenized_sentences;

    def createIndex(self, tokenized_sentences, vocabulary_size=_VOCABULARY_SIZE):
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

        # print >> sys.stderr, word_freq.r_Nr();
        # print >> sys.stderr, word_freq.B();

        if (vocabulary_size == -1):
            vocab = word_freq.keys();
            index_to_word = [x for x in vocab]
        else:
            vocab = word_freq.most_common(vocabulary_size - 1)
            index_to_word = [x[0] for x in vocab]
        index_to_word.append(_UNKNOWN_TOKEN)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        return word_to_index;

    def extract_seq_targets(self, features, times, rumourIds, labels, makeTarget=True, retunr_indices=False,
                            binary=False):
        tempSeqs = {};

        index = -1;
        for r in rumourIds:
            index += 1;
            if r in tempSeqs:
                seq = tempSeqs[r];
            else:
                seq = [];
                tempSeqs[r] = seq;
            seq.append((features[index], times[index], index));

        seqs = [];
        if binary:
            targets_s = [];
            targets_d = [];
            targets_q = [];
            targets_c = [];
        else:
            targets = [];
        indices = [];
        for key, value in tempSeqs.iteritems():
            indices.append(list(map(lambda tup: tup[2], tempSeqs[key])));
            tempSeqs[key].sort(key=lambda tup: tup[1]);
            seqs.append(list(map(lambda tup: tup[0], tempSeqs[key])));
            if makeTarget:
                if binary:
                    targets_s.append(list(map(lambda tup: 1 if int(labels[tup[2]]) == 0 else 0, tempSeqs[key])));
                    targets_d.append(list(map(lambda tup: 1 if int(labels[tup[2]]) == 1 else 0, tempSeqs[key])));
                    targets_q.append(list(map(lambda tup: 1 if int(labels[tup[2]]) == 2 else 0, tempSeqs[key])));
                    targets_c.append(list(map(lambda tup: 1 if int(labels[tup[2]]) == 3 else 0, tempSeqs[key])));
                else:
                    targets.append(
                        list(map(lambda tup: [1 if i == labels[tup[2]] else 0 for i in xrange(4)], tempSeqs[key])));

        if makeTarget:
            if retunr_indices:
                if binary:
                    return seqs, targets_s, targets_d, targets_q, targets_c, indices
                else:
                    return seqs, targets, indices;
            else:
                if binary:
                    return seqs, targets_s, targets_d, targets_q, targets_c
                else:
                    return seqs, targets;
        else:
            if retunr_indices:
                return seqs, indices;
            else:
                return seqs;

    def extract_threads_targets(self, features, tweetIds, infectedTweetIds, labels, makeTarget=True, return_full=False):
        trees = [];

        for ind, tweetId in enumerate(tweetIds):
            infectedTweetId = infectedTweetIds[ind];
            infectedTweet = None
            for tree in trees:
                infectedTweet = tree.find(infectedTweetId);
                if infectedTweet is not None:
                    break;
            if infectedTweet is None:
                if makeTarget:
                    auxilary = (features[list(infectedTweetIds).index(infectedTweetId)],
                                [1 if i == labels[list(infectedTweetIds).index(infectedTweetId)] else 0 for i in xrange(4)])
                else:
                    auxilary = (features[list(infectedTweetIds).index(infectedTweetId)], []);
                infectedTweet = Tree(data=infectedTweetId, auxilaries=auxilary);
                trees.append(infectedTweet)
            if infectedTweetId != tweetId:
                tweet = None;
                for tree in trees:
                    tweet = tree.find(tweetId);
                    if tweet is not None:
                        break;
                if tweet is not None:
                    infectedTweet.addChildNode(tweet)
                else:
                    if makeTarget:
                        auxilary = (features[list(tweetIds).index(tweetId)],
                                    [1 if i == labels[list(tweetIds).index(tweetId)] else 0 for i in
                                     xrange(4)])
                    else:
                        auxilary = (features[list(tweetIds).index(tweetId)], []);
                    infectedTweet.addChild(tweetId, auxilaries=auxilary);

        tweetTree = Tree();
        for tree in trees:
            root = tree.getRoot()
            if root is tree:
                tweetTree.addChildNode(root);

        threads = tweetTree.extractThreads();

        if makeTarget:
            return map(lambda thread: [e.auxilaries[0] for e in thread], threads), map(
                lambda thread: [e.auxilaries[1] for e in thread], threads);
        else:
            if return_full:
                return threads;
            else:
                return map(lambda thread: [e.auxilaries[0] for e in thread], threads);

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
        return self.predict_test(
            testN,
            testtimes,
            testinfected_vec,
            testinfecting_vec,
            testeventmemes,
            testW,
            testT,
            testnode_vec)

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
            rememberHistory=False
    ):
        predictednode_vec = np.zeros((len(testtimes),), dtype=int);

        tweetTexts = tweet.readTweetContents(map(int, testeventmemes), map(int, testinfecting_vec))
        # features = self.convertTextToFeature(tweetTexts);
        features = self.doc2vec.infer_test_vec(tweetTexts);

        if self.useTree:
            predictions = {};
            x_test = self.extract_threads_targets(features, testinfecting_vec, testinfected_vec, [], makeTarget=False, return_full=True);
            for thread in x_test:
                for point in thread:
                    feature = point.auxilaries[0];
                    tweetId = point.data;
                    ind = list(testinfecting_vec).index(int(tweetId));
                    label = self.model.predict_classes(np.asarray([[list(feature)]]), batch_size=1)[0];
                    if ind in predictions.keys():
                        predictions[ind].append(label);
                    else:
                        predictions[ind] = [label];
                self.model.reset_states()
            assert len(predictions.keys()) == len(testtimes)
            for ind in predictions.keys():
                predictednode_vec[ind] = np.argmax(np.bincount(np.asarray(predictions[ind])));

        else:
            x_test, indices = self.extract_seq_targets(features, testtimes, testeventmemes, [], makeTarget=False,
                                                       retunr_indices=True);
            # x_test = x_test[0];
            # indices = indices[0]

            for i in xrange(len(x_test)):
                for j in xrange(len(x_test[i])):
                    if self.binary:
                        s_score = self.model_s.predict_on_batch(np.asarray([[list(x_test[i][j])]]))[0];
                        d_score = self.model_s.predict_on_batch(np.asarray([[list(x_test[i][j])]]))[0];
                        q_score = self.model_s.predict_on_batch(np.asarray([[list(x_test[i][j])]]))[0];
                        c_score = self.model_s.predict_on_batch(np.asarray([[list(x_test[i][j])]]))[0];

                        scores = [s_score, d_score, q_score, c_score];
                        max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))

                        # if len(x_test) == 1:
                        predictednode_vec[indices[i][j]] = max_index;
                    else:
                        # if len(x_test) == 1:
                        predictednode_vec[indices[i][j]] = \
                                self.model.predict_classes(np.asarray([[list(x_test[i][j])]]), batch_size=1)[0];
                        if(rememberHistory):
                            self.histories[testinfecting_vec[indices[i][j]]] = list(K.get_value(self.model.layers[0].states[1])[0])


                if self.binary:
                    self.model_s.reset_states()
                    self.model_d.reset_states()
                    self.model_q.reset_states()
                    self.model_c.reset_states()
                else:
                    self.model.reset_states()

        if rememberHistory is not True:
            print >> sys.stderr, predictednode_vec;
            print >> sys.stderr, testnode_vec;

        return predictednode_vec

        # return np.zeros((len(testtimes, ))) + 3;

    def getHistoryState(self, tweetId):
        if tweetId in self.histories.keys():
            return self.histories[tweetId];
        else:
            raise ValueError('No history for the ', str(tweetId), ' tweet ID!')

    def resample(self):
        label_c_indices = self.node_vec == 3
        label_other_indices = list(np.invert(label_c_indices));
        node_vec_c = list(compress(self.node_vec, label_c_indices));
        node_vec_other = list(compress(self.node_vec, label_other_indices));
        etimes_c = list(compress(self.etimes, label_c_indices));
        etimes_other = list(compress(self.etimes, label_other_indices));
        ememes_c = list(compress(self.ememes, label_c_indices));
        ememes_other = list(compress(self.ememes, label_other_indices));
        infecting_vec_c = list(compress(self.infecting_vec, label_c_indices));
        infecting_vec_other = list(compress(self.infecting_vec, label_other_indices));
        infected_vec_c = list(compress(self.infected_vec, label_c_indices));
        infected_vec_other = list(compress(self.infected_vec, label_other_indices));

        # print >> sys.stderr, len(node_vec_c)
        # print >> sys.stderr, len(node_vec_other)

        selected_c_indices = np.random.choice([True, False], size=(len(node_vec_c,)), p=[1./2, 1./2])
        selected_node_vec_c = list(compress(node_vec_c, selected_c_indices));
        selected_etimes_c = list(compress(etimes_c, selected_c_indices));
        selected_ememes_c = list(compress(ememes_c, selected_c_indices));
        selected_infecting_vec_c = list(compress(infecting_vec_c, selected_c_indices));
        selected_infected_vec_c = list(compress(infected_vec_c, selected_c_indices));

        # print >> sys.stderr, len(selected_node_vec_c)

        node_vec_other.extend(selected_node_vec_c)
        etimes_other.extend(selected_etimes_c)
        ememes_other.extend(selected_ememes_c)
        infected_vec_other.extend(selected_infected_vec_c)
        infecting_vec_other.extend(selected_infecting_vec_c)

        self.node_vec = node_vec_other;
        self.etimes = etimes_other;
        self.ememes = ememes_other;
        self.infecting_vec = infecting_vec_other;
        self.infected_vec = infected_vec_other;

        # print >> sys.stderr, len(self.node_vec)
        # print >> sys.stderr, len(self.etimes)
        # print >> sys.stderr, len(self.ememes)
        # print >> sys.stderr, len(self.infected_vec)
        # print >> sys.stderr, len(self.infecting_vec)


