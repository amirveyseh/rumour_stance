import numpy as np
import sys


def foldsplitter(taskcolumn, train_set_sizes):
    folds = sorted(list(set(taskcolumn)))
    for fold in folds:
        for train_set_size in train_set_sizes:
            testfold2train = taskcolumn == fold
            cnt = 0
            for (i, x) in enumerate(testfold2train):
                if testfold2train[i]:
                    cnt += 1
                    if cnt > train_set_size:
                        testfold2train[i] = False
            remaining_train = taskcolumn != fold
            x = np.logical_or.reduce([testfold2train, remaining_train])
            # x = np.logical_or.reduce([np.logical_not(testfold2train), remaining_train])

            yield (x, np.logical_not(x))
            # yield (x, np.logical_not(remaining_train))


def CVsplitter(taskcolumn, K):

    tasks = sorted(list(set(taskcolumn)))
    tasks_splitted = [[] for _ in range(K)]
    for (ind, task) in enumerate(tasks):
        tasks_splitted[ind % K].append(task)

    for fold in range(K):
        print 'fold:', fold, 'testtasks:', tasks_splitted[fold]
        test = np.logical_or.reduce([taskcolumn == taskid for taskid in
                                    tasks_splitted[fold]])

        yield (np.logical_not(test), test)


