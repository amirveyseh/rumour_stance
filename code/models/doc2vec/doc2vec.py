
import gensim.models as g
import codecs, nltk, os, sys
from nltk.corpus import stopwords
reload(sys)
sys.setdefaultencoding('utf-8')

#doc2vec parameters
_VECTOR_SIZE = 128
_WINDOW_SIZE = 5
_MIN_COUNT = 1
_SAMPLING_THRESHOLD = 1e-5
_NEGATIVE_SIZE = 5
_TRAIN_EPOCH = 100
_DM = 0 #0 = dbow; 1 = dmpv
_WORKER_COUNT = 1 #number of parallel processes

# inference hyper-parameters
_START_ALPHA = 0.01
_INFER_EPOCH = 1000

class Doc2Vec():
    def __init__(self, train_corpus='train_corpus.txt', infer_corpus='infer_corpus.txt'):
        self.train_corpus = train_corpus;
        self.infer_corpus = infer_corpus;

    def train(self):
        docs = g.doc2vec.TaggedLineDocument(self.train_corpus)
        model = g.Doc2Vec(docs, size=_VECTOR_SIZE, window=_WINDOW_SIZE, min_count=_MIN_COUNT, sample=_SAMPLING_THRESHOLD,
                          workers=_WORKER_COUNT, hs=0, dm=_DM, negative=_NEGATIVE_SIZE, dbow_words=1, dm_concat=1,
                          iter=_TRAIN_EPOCH)
        self.model = model;
        self.clear_corpus(train=True)
        return model;

    def infer_vector(self):
        infer_vec = [];
        test_docs = [x.strip().split() for x in codecs.open(self.infer_corpus, "r", "utf-8").readlines()]
        for d in test_docs:
            infer_vec.append(self.model.infer_vector(d, alpha=_START_ALPHA, steps=_INFER_EPOCH))
        self.clear_corpus(train=False)
        return infer_vec;

    def make_corpus(self, texts, train=True):
        texts = self.tokenize_texts(texts);
        if train:
            output_file = self.train_corpus;
        else:
            output_file = self.infer_corpus;
        output = open(output_file, "w")
        for text in texts:
            output.write(" ".join([str(x) for x in text])+"\n");
        output.flush();
        output.close();

    def tokenize_texts(self, texts):
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

    def clear_corpus(self, train=True):
        if train:
            os.remove(self.train_corpus);
        else:
            os.remove(self.infer_corpus);

    def extract_train_vec(self, texts):
        self.make_corpus(texts,train=True);
        self.train();
        return self.model.docvecs

    def infer_test_vec(self, texts):
        self.make_corpus(texts,train=False);
        return self.infer_vector();