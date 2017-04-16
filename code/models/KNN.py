import tweet, numpy as np, sys, time;
from sklearn.neighbors import KNeighborsClassifier
from doc2vec.doc2vec import Doc2Vec



class KNN():
    def __init__(self, rumourIds=[], tweetIds=[], infected_vec=[], labels=[], times=[], topic=-1):
        self.rumourIds = rumourIds;
        self.tweetIds = tweetIds;
        self.infected_vec = infected_vec;
        self.labels = labels;
        self.times = times;
        self.topic = topic;
        self.doc2vec = Doc2Vec();
        self.model = KNeighborsClassifier();

    def train(self):
        print >> sys.stderr, 'KNN Training...'
        t1 = time.time()

        x_train, y_train = self.makeFeaturedData(self.rumourIds, self.tweetIds, labels=self.labels);
        self.model.fit(x_train,y_train);

        t2 = time.time()
        print >> sys.stderr, "Training Time: %f seconds" % ((t2 - t1) * 1.)


    def makeFeaturedData(self, rumourIds, tweetIds, labels=[], makeTarget=True):
        # x_train = tweet.extractVocabFeature(rumourIds, tweetIds);
        x_train = self.doc2vec.extract_train_vec(tweet.readTweetContents(map(int, rumourIds), map(int, tweetIds), self.topic));
        if makeTarget:
            y_train = labels;
            return x_train, y_train
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

        x_test = self.makeFeaturedData(testeventmemes, testinfecting_vec, makeTarget=False)

        predictednode_vec = self.model.predict(x_test);

        print >> sys.stderr, predictednode_vec;
        print >> sys.stderr, testnode_vec;

        return predictednode_vec