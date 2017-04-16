from models.baselines import LearnIntensity
# from models.hpseq_baselines import RNNLM, LearnIntensity
# from models.seq2seqRNN import seq2seqRNN
# from models.ContextLM import ContextLM
from models.ContexLMJoint import ContextLMJoint
from models.KNN import KNN


def get_methods():
    model_constructors = [
        #               ('RNNLM', lambda etimes, node_vec, eventmemes, \
        #      infected_vec, infecting_vec, W, T, V, D, topic: RNNLM(
        # etimes,
        # node_vec,
        # eventmemes,
        # infected_vec,
        # infecting_vec,
        # W,
        # T,
        # V,
        # D,
        # [],
        # ITERATIONS,
        # verbose=False,
        # topic=topic
        # )),
        ('LearnIntensity', lambda etimes, \
                                  node_vec, eventmemes, infected_vec, \
                                  infecting_vec, W, T, V, D, topic: LearnIntensity(
            etimes,
            node_vec,
            eventmemes,
            infected_vec,
            infecting_vec,
            W,
            T,
            V,
            D,
            [],
            0,
            topic=topic
        )),
        #               ('seq2seqRNN', lambda etimes, \
        #               node_vec, eventmemes, infected_vec, \
        #               infecting_vec, W, T, V, D, topic=topic: seq2seqRNN(
        # times=etimes,
        # labels=node_vec,
        # rumourIds=eventmemes,
        # infected_vec=infecting_vec,
        # tweetIds=infected_vec,
        # topic=topic
        # )), ('ContextLM', lambda etimes, \
        #               node_vec, eventmemes, infected_vec, \
        #               infecting_vec, W, T, V, D, topic: ContextLM(
        # times=etimes,
        # labels=node_vec,
        # rumourIds=eventmemes,
        # infected_vec=infected_vec,
        # tweetIds=infecting_vec,
        # topic=topic
        # )),
        ('KNN', lambda etimes, \
                       node_vec, eventmemes, infected_vec, \
                       infecting_vec, W, T, V, D, topic: KNN(
            times=etimes,
            labels=node_vec,
            rumourIds=eventmemes,
            infected_vec=infected_vec,
            tweetIds=infecting_vec,
            topic=topic
        )),
        ('ContextLMJoint', lambda etimes, \
                                  node_vec, eventmemes, infected_vec, \
                                  infecting_vec, W, T, V, D, topic: ContextLMJoint(
            times=etimes,
            labels=node_vec,
            rumourIds=eventmemes,
            infected_vec=infecting_vec,
            tweetIds=infected_vec,
            topic=topic
        ))
    ]
    return model_constructors


def get_methodnames():
    return map(lambda x: x[0], get_methods())
