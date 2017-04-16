import tweetUtils;
import sys, numpy as np, nltk, itertools;
from nltk.corpus import stopwords

_VOCABULARY_SIZE = 800;
_UNKNOWN_TOKEN = "UNKNOWN_TOKEN";
TOPIC = -1;


def get_tweet_content(topic, rumourId, tweetId):
    tweet = tweetUtils.json_parse(topic + '/' + rumourId + '/' + tweetId + '.json');
    return tweet['text'];


def readTweetContents(rumourIds, tweetIds, topic):
    paths = {
        'sydney': '../data/processedData/sydney',
        'ferguson': '../data/processedData/fergousen',
        'charlie': '../data/processedData/charlie',
        'ottawa': '../data/processedData/ottawa',
        'all': '../data/processedData/all'
    };
    tweets = [];
    for i in xrange(len(rumourIds)):
        tweets.append(get_tweet_content(paths[topic], str(rumourIds[i]), str(tweetIds[i])));
    return tweets;


def extractVocabFeature(rumourIds, tweetIds, vocabSize=_VOCABULARY_SIZE):
    tweetTexts = readTweetContents(map(int, rumourIds), map(int, tweetIds));
    features = convertTextToFeature(tweetTexts, vocabSize=vocabSize);
    return features;


def convertTextToFeature(texts, vocabSize=_VOCABULARY_SIZE):
    processedTexts = preprocessTexts(texts, vocabSize=vocabSize);
    features = extractFeatures(processedTexts, vocabSize=vocabSize);
    return features;


def preprocessTexts(texts, vocabSize=_VOCABULARY_SIZE, remove_stop_words=True, mask_zero=False):
    tokenized_sentences = tokenize(texts, remove_stop_words=remove_stop_words);
    word_to_index = createIndex(tokenized_sentences, vocabulary_size=vocabSize);
    tokenized_sentences = normalizeText(tokenized_sentences, word_to_index);
    if mask_zero:
        shift = 1;
    else:
        shift = 0;
    processedTexts = np.asarray([[word_to_index[w]+shift for w in sent] for sent in tokenized_sentences])
    return processedTexts;


def tokenize(texts, remove_stop_words=True):
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
    if remove_stop_words:
        for i, x in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [word for word in x if
                                  (word not in stopwords.words('english')) or (word in ['no', 'not'])]
    return tokenized_sentences;


def normalizeText(tokenized_sentences, word_to_index):
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else _UNKNOWN_TOKEN for w in sent]
    return tokenized_sentences;


def createIndex(tokenized_sentences, vocabulary_size=_VOCABULARY_SIZE):
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


def extractFeatures(texts, vocabSize=_VOCABULARY_SIZE):
    features = [];
    for text in texts:
        feature = np.zeros((vocabSize), dtype=int);
        for token in text:
            feature[token] = 1;
        features.append(feature);
    return features;
