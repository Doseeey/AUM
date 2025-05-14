import numpy as np

def accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    return np.mean(y_true.ravel()==y_pred.ravel())

def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    tp = np.zeros(len(unique_labels))
    fp = np.zeros(len(unique_labels))

    for i in unique_labels: #labels are [0, 1, 2]
        tp[i] = np.sum((y_true == i) & (y_pred == i))
        fp[i] = np.sum((y_true != i) & (y_pred == i))


    if average == 'macro':
        return np.mean(tp/(tp+fp))
    elif average == 'micro':
        return np.sum(tp)/(np.sum(tp)+np.sum(fp))
    else:
        return 0

def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    tp = np.zeros(len(unique_labels))
    fn = np.zeros(len(unique_labels))

    for i in unique_labels: #labels are [0, 1, 2]
        tp[i] = np.sum((y_true == i) & (y_pred == i))
        fn[i] = np.sum((y_true == i) & (y_pred != i))


    if average == 'macro':
        return np.mean(tp/(tp+fn))
    elif average == 'micro':
        return np.sum(tp)/(np.sum(tp)+np.sum(fn))
    else:
        return 0

def f1_score(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""

    p = precision(y_true, y_pred, average=average)
    r = recall(y_true, y_pred, average=average)
    return (2*p*r)/(p+r)