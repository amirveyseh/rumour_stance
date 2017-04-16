topics = [];

ms = [];
ms.append((63.43, 19.41, 'Majority'));
ms.append((67.26, 37.74, 'SVM'));
ms.append((65.04, 42.24, 'GP'));
ms.append((51.60, 41.51, 'Lang.'));
ms.append((62.01, 38.56, 'NB'));
ms.append((67.44, 35.74, 'CRF'));
ms.append((68.59, 32.49, 'HP Approx.'));
ms.append((62.99, 39.45, 'HP Grad.'));
ms.append((68.20, 43.73, 'LM_OvA'));
ms.append((69.47, 32.05, 'seq2seqRNN'));

topics.append((ms, 'Sydney'));

ms = [];
ms.append((67.53, 20.15, 'Majority'));
ms.append((69.90, 35.11, 'SVM'));
ms.append((70.66, 44.09, 'GP'));
ms.append((63.44, 42.84, 'Lang.'));
ms.append((70.18, 39.69, 'NB'));
ms.append((71.89, 40.12, 'CRF'));
ms.append((72.93, 32.56, 'HP Approx.'));
ms.append((71.79, 41.91, 'HP Grad.'));
ms.append((70.40, 37.96, 'LM_OvA'));
ms.append((71.90, 31.38, 'seq2seqRNN'));

topics.append((ms, 'Charlie'));

ms = [];
ms.append((66.86, 20.04, 'Majority'));
ms.append((66.86, 20.04, 'SVM'));
ms.append((64.31, 32.90, 'GP'));
ms.append((49.56, 34.35, 'Lang.'));
ms.append((62.05, 31.29, 'NB'));
ms.append((67.35, 28.11, 'CRF'));
ms.append((68.44, 25.99, 'HP Approx.'));
ms.append((63.23, 33.14, 'HP Grad.'));
ms.append((62.08, 30.50, 'LM_OvA'));
ms.append((69.28, 28.77, 'seq2seqRNN'));

topics.append((ms, 'Ferguson'));

ms = [];
ms.append((61.51, 19.04, 'Majority'));
ms.append((64.58, 35.39, 'SVM'));
ms.append((62.28, 42.41, 'GP'));
ms.append((53.20, 42.66, 'Lang.'));
ms.append((61.76, 40.64, 'NB'));
ms.append((64.58, 33.07, 'CRF'));
ms.append((67.77, 32.29, 'HP Approx.'));
ms.append((63.43, 42.40, 'HP Grad.'));
ms.append((64.35, 37.89, 'LM_OvA'));
ms.append((66.67, 31.72, 'seq2seqRNN'));
ms.append((65.25, 29.69, 'treeRNN'));

topics.append((ms, 'Ottawa'));


for t in topics:
    print '*********', t[1], '************'
    for item in t[0]:
        result = (2 * item[0] * item[1]) / (item[0] + item[1]);
        print item[2], result;

