import sys
from experiment.experiment_stance_classif import ExperimentStanceClassification
import sklearn.metrics
from header_methods import get_methods
from experiment.utils import foldsplitter

if __name__ == '__main__':
    FOLDTORUN = int(sys.argv[1])
    methodname = sys.argv[2]
    train_set_ratio = int(sys.argv[3])
    fname_data = sys.argv[4]
    topic = sys.argv[5]
    DO_TRAIN = True
    DO_PLOT = False

    metrics = [('ACCURACY', sklearn.metrics.accuracy_score)]

    model_constructors = get_methods()
    model_constructors = filter(lambda x: x[0] == methodname,
                                    model_constructors)

    exp = ExperimentStanceClassification(
        fname_data,
        model_constructors,
        train_set_ratio,
        metrics,
        foldsplitter,
        FOLDTORUN=FOLDTORUN,
        topic=topic
        )
    exp.build_models(train=DO_TRAIN)

    exp.evaluate()
    exp.summarize()