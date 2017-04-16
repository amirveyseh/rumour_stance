import tweetUtils;


def get_rumours_id(topic):
    return tweetUtils.get_immediate_subdirectories(topic)


def get_mapped_rumour_ids(topic):
    rumours = get_rumours_id(topic);
    rumours = sorted(rumours);
    map = {};
    i = -1;
    for r in rumours:
        i += 1;
        map[r] = i;
    return map;


def get_mapped_tweet_ids(topic):
    rumours = get_rumours_id(topic);
    tweets = [];
    for r in rumours:
        tweets.extend(get_tweet_ids(topic, r));
    tweets = sorted(tweets);
    map = {};
    i = -1;
    for t in tweets:
        i += 1;
        map[t] = i;
    return map;


def get_mapped_rumour_id(topic, rumourId):
    ids = get_mapped_rumour_ids(topic);
    return ids[rumourId];


def get_mapped_tweet_id(topic, tweetId):
    ids = get_mapped_tweet_ids(topic);
    return ids[tweetId];


def get_tweet_ids(topic, rumour):
    replies = tweetUtils.get_files(topic + '/' + rumour + '/replies');
    sources = tweetUtils.get_files(topic + '/' + rumour + '/source-tweet');
    replies.extend(sources)
    tweets = [];
    for r in replies:
        tweets.append(r.replace('.json', ''));
    return tweets;


def extract_rumour_seqs(rumourIds, tweetIds, times):
    tempSeqs = {};
    for index, r in enumerate(rumourIds):
        if r in tempSeqs:
            seq = tempSeqs[r];
        else:
            seq = [];
            tempSeqs[r] = seq;
        seq.append((tweetIds[index], times[index]));
    seqs = [];
    for key in sorted(tempSeqs.iterkeys()):
        tempSeqs[key].sort(key=lambda tup: tup[1]);
        seqs.append(list(map(lambda tup: tup[0], tempSeqs[key])));
    return seqs;

