import numpy as np
import tweet
import rumours
import sys
# from LM_with_RNN import LM_With_RNN;
import operator
import time
from rnn_minibatch import MetaRNN
# from kerasRNN import kerasRNN

# class RNNLM(HPSeq):
#
#     def readTweetContents(self, rumourIds, tweetIds):
#         paths = [];
#         paths.append(
#             '/home/amir/Desktop/research/seqhawkes/data/processedData/sydney');
#         paths.append('/home/amir/Desktop/research/seqhawkes/data/processedData/charlie');
#         paths.append('/home/amir/Desktop/research/seqhawkes/data/processedData/ottawa');
#         paths.append('/home/amir/Desktop/research/seqhawkes/data/processedData/fergousen');
#
#         tweets = [];
#
#         for i in xrange(len(rumourIds)):
#             tweets.append(tweet.get_tweet_content(paths[0],str(rumourIds[i]),str(tweetIds[i])));
#         return tweets;
#
#     def train(self):
#         print >> sys.stderr, 'Training!'
#         t1 = time.time()
#         s_indices = self.node_vec == 0;
#         d_indices = self.node_vec == 1;
#         q_indices = self.node_vec == 2;
#         c_indices = self.node_vec == 3;
#         s_tweetTexts = self.readTweetContents(map(int, self.ememes[s_indices]), map(int, self.infecting_vec[s_indices]))
#         d_tweetTexts = self.readTweetContents(map(int, self.ememes[d_indices]), map(int, self.infecting_vec[d_indices]))
#         q_tweetTexts = self.readTweetContents(map(int, self.ememes[q_indices]), map(int, self.infecting_vec[q_indices]))
#         c_tweetTexts = self.readTweetContents(map(int, self.ememes[c_indices]), map(int, self.infecting_vec[c_indices]))
#         self.lm_s = LM_With_RNN(s_tweetTexts);
#         self.lm_d = LM_With_RNN(d_tweetTexts);
#         self.lm_q = LM_With_RNN(q_tweetTexts);
#         self.lm_c = LM_With_RNN(c_tweetTexts);
#         t2 = time.time()
#         print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.)
#
#
#     def evaluate_stance(
#         self,
#         testN,
#         testtimes,
#         testinfecting_vec,
#         testinfected_vec,
#         testeventmemes,
#         testW,
#         testT,
#         ):
#         predictednode_vec = [None for _ in xrange(len(testtimes))]
#         tweetTexts = self.readTweetContents(map(int, testeventmemes), map(int, testinfecting_vec))
#         index = -1;
#         for tt in tweetTexts:
#             index += 1;
#
#             s_score = self.lm_s.calculate_score(tt);
#             d_score = self.lm_d.calculate_score(tt);
#             q_score = self.lm_q.calculate_score(tt);
#             c_score = self.lm_c.calculate_score(tt);
#
#             scores = [s_score, d_score, q_score, c_score];
#             min_index, min_value = min(enumerate(scores), key=operator.itemgetter(1))
#
#             predictednode_vec[index] = min_index;
#
#         return predictednode_vec


class LearnIntensity:
    def __init__(self, *args):
        print >> sys.stderr, 'debug for LearnIntensity ';

        self.node_vec = args[1];
        # print >> sys.stderr, len(self.node_vec);
        self.ememes = args[2];
        self.infecting_vec = args[4];


        t1 = time.time()
        # print >> sys.stderr, 'debug - class: ', args[1];
        # print >> sys.stderr, 'debug - rumourID: ', args[2];
        # print >> sys.stderr, 'debug - tokens: ', args[5];

        classes = args[1];
        rumourIDs = args[2];
        tokens = args[5];

        # numberOfinstancePerRumour = np.zeros((int(max(rumourIDs))+1,), dtype=np.int);
        # for i in xrange(len(rumourIDs)):
        #     numberOfinstancePerRumour[int(rumourIDs[i])] += 1;
        # print >> sys.stderr, 'debug - numberOfinstancePerRumour: ', numberOfinstancePerRumour;

        n_in = len(tokens[0]);
        n_hidden = 500;
        n_out = 4;
        n_epochs = 500;
        batch_size = 100;

        # print >> sys.stderr, "vocab size: ", n_in;

        # seq = np.zeros((int(max(rumourIDs))+1,), dtype=np.int);
        # targets = np.zeros((int(max(rumourIDs))+1,), dtype=np.int)-1;

        seq = [None] * (int(max(rumourIDs)) + 1);
        targets = [None] * (int(max(rumourIDs)) + 1);
        for i in xrange(len(rumourIDs)):
            seq[int(rumourIDs[i])] = [];
            targets[int(rumourIDs[i])] = [];

        for i in xrange(len(rumourIDs)):
            seq[int(rumourIDs[i])] += [tokens[i]];
            targets[int(rumourIDs[i])] += [int(classes[i])];
            # print >> sys.stderr, "seq: ", seq[int(rumourIDs[i])];
            # print >> sys.stderr, "token: ", tokens[i];

        seq = [x for x in seq if x is not None];
        targets = [x for x in targets if x is not None];

        # print >> sys.stderr, "seq: ", seq;
        # print >> sys.stderr, "targets: ", targets;

        cutOff = 14;
        ls = [];
        for i in xrange(len(seq)):
            if seq[i] is None:
                print >> sys.stderr, "none", i;
                sys.exit();
            l = len(seq[i]);
            ls.append(l);
            if(l < cutOff):
                for j in xrange(cutOff-l):
                    seq[i] += [seq[i][-1]];
                    targets[i] += [targets[i][-1]];
            elif(l > cutOff):
                for j in xrange(l-cutOff):
                    seq[i][j+cutOff] = None;
                    targets[i][j+cutOff] = None;
        # print >> sys.stderr, 'average seq length: ', np.average(ls);
        # print >> sys.stderr, 'min seq length: ', np.min(ls);
        # print >> sys.stderr, 'max seq length: ', np.max(ls);
        # print >> sys.stderr, 'seq length: ', ls;

        i = -1;
        for x in seq:
            i += 1;
            seq[i] = [y for y in x if y is not None];
        i = -1;
        for x in targets:
            i += 1;
            targets[i] = [y for y in x if y is not None];


        # model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
        #                 learning_rate=0.001, learning_rate_decay=0.999,
        #                 n_epochs=n_epochs, activation='tanh',
        #                 output_type='softmax', use_symbolic_softmax=False)
        #
        # model.fit(seq, targets,validation_frequency=1000)

        ###### mini-batch
        seq = np.transpose(seq, axes=(1, 0, 2)).tolist()
        targets = np.transpose(targets, axes=(1, 0)).tolist()

        model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                        learning_rate=0.005, learning_rate_decay=0.999,
                        n_epochs=n_epochs, batch_size=batch_size, activation='tanh',
                        output_type='softmax')

        model.fit(seq, targets, validate_every=10, compute_zero_one=True,
                  optimizer='sgd')

        self.model = model;
        t2 = time.time()
        print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.)
        print >> sys.stderr, "Training Done!"

    def train(self):
        # t1 = time.time();
        # # tweetTexts = tweet.readTweetContents(map(int, self.ememes), map(int, self.infecting_vec));
        # # self.model = kerasRNN(texts=tweetTexts, labels=self.node_vec, static=True);
        # # x_train = self.model.preprocess_text(texts=tweetTexts, createIndex=True);
        # # ls = [];
        # # for s in x_train:
        # #     ls.append(len(s));
        # # print >> sys.stderr, 'max : ', np.max(ls);
        #
        #
        # tweetTexts = tweet.readTweetContents(map(int, self.ememes), map(int, self.infecting_vec));
        # self.lm_s = kerasRNN(texts=tweetTexts, labels=self.node_vec, binary=True, targetClass=0);
        # self.lm_d = kerasRNN(texts=tweetTexts, labels=self.node_vec, binary=True, targetClass=1);
        # self.lm_q = kerasRNN(texts=tweetTexts, labels=self.node_vec, binary=True, targetClass=2);
        # self.lm_c = kerasRNN(texts=tweetTexts, labels=self.node_vec, binary=True, targetClass=3);
        #
        # t2 = time.time();
        # print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.);
        print >> sys.stderr, "Training Done!"

    def evaluate_stance(
            self,
            testN,
            testtimes,
            testinfecting_vec,
            testinfected_vec,
            testeventmemes,
            testW,
            testT,
            testnode_vec
    ):
        # print >> sys.stderr, 'debug - testN: ', testN;
        # print >> sys.stderr, 'debug - testtimes: ', testtimes;
        # print >> sys.stderr, 'debug - testinfecting_vec: ', testinfecting_vec;
        # print >> sys.stderr, 'debug - testinfected_vec: ', testinfected_vec;
        # print >> sys.stderr, 'debug - testeventmemes: ', testeventmemes;
        # print >> sys.stderr, 'debug - testW: ', testW;
        # print >> sys.stderr, 'debug - testT: ', testT;

        rumourIDs = testeventmemes;
        tokens = testW;

        seq = [None] * (int(max(rumourIDs)) + 1);

        for i in xrange(len(rumourIDs)):
            seq[int(rumourIDs[i])] = [];

        for i in xrange(len(rumourIDs)):
            seq[int(rumourIDs[i])] += [tokens[i]];

        seq = [x for x in seq if x is not None];

        if (len(seq) > 1):
            print >> sys.stderr, '************* Exiting script because there is something wrong in evaluation! ************** ';
            sys.exit();

        for i in xrange(len(seq)):
            prediction_vec = self.model.predict(np.asarray(seq[i])[:, np.newaxis]);
            prediction_vec = prediction_vec.reshape((prediction_vec.shape[0],));
            # prediction_vec = self.model.predict(seq[i]);
            print >> sys.stderr, prediction_vec;
            print >> sys.stderr, testnode_vec;
        return prediction_vec;


        # predictednode_vec = [None for _ in xrange(len(testtimes))]
        # tweetTexts = tweet.readTweetContents(map(int, testeventmemes), map(int, testinfecting_vec))
        #
        # index = -1;
        # for tt in tweetTexts:
        #     index += 1;
        #     predictednode_vec[index] = self.model.predict(tt);
        #
        # return predictednode_vec

        # predictednode_vec = [None for _ in xrange(len(testtimes))]
        # tweetTexts = tweet.readTweetContents(map(int, testeventmemes), map(int, testinfecting_vec))
        # index = -1;
        # for tt in tweetTexts:
        #     index += 1;
        #
        #     s_score = self.lm_s.predict(tt, score=True);
        #     d_score = self.lm_d.predict(tt, score=True);
        #     q_score = self.lm_q.predict(tt, score=True);
        #     c_score = self.lm_c.predict(tt, score=True);
        #
        #     scores = [s_score, d_score, q_score, c_score];
        #     max_index, max_value = max(enumerate(scores), key=operator.itemgetter(1))
        #
        #     predictednode_vec[index] = max_index;
        #     # predictednode_vec[index] = 3;
        #
        # return predictednode_vec

        # return np.zeros((len(testtimes,)))+3;

