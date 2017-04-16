import numpy as np
import sys


def load(fname):

    with open(fname) as f:
        for l in f:
            l = l.split()
            memeid = int(l[0])
            eventid = int(l[1])
            infectingid = int(l[2])
            nodeid = int(l[3])
            time = float(l[4])
            wordslen = int(l[5])
            words = {}
            for wordidx in xrange(wordslen):
                word = l[6 + wordidx]
                word = map(int, word.split(':'))
                words[word[0]] = word[1]
            yield (
                memeid,
                eventid,
                infectingid,
                nodeid,
                time,
                words
                )



def loadX(fname):

    events = []
    words_keys = set()
    for e in load(fname):
        events.append(e)
        words_keys = words_keys | set(e[5].keys())
    words_keys = sorted(list(words_keys))
    for (eidx, e) in enumerate(events):
        events[eidx] = list(e[:5]) + [e[5].get(word_key, 0)
                for word_key in words_keys]
    X = np.vstack(events)
    return (X, words_keys)


