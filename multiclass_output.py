import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics

plt.rcParams["figure.figsize"] = (10,5)


def print_conf_matrix(m, labels=None):
    if not isinstance(m, pd.DataFrame):
        m = pd.DataFrame(m, labels, labels)
    print(sum(m.values.flatten()))
    sn.set(font_scale=1)
    ax = sn.heatmap(m.round(2), annot=True, annot_kws={"size": 12}, fmt='g')
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)
    plt.show()

def get_avg_cm(preds, normalize=None):
    res = np.sum([metrics.confusion_matrix(preds[i][0], preds[i][1], normalize=normalize) for i in range(len(preds))], axis=(0))
    return res/len(preds) if normalize else res

def get_avg_cm_df(preds, labels, normalize=None):
    return pd.DataFrame(get_avg_cm(preds, normalize=normalize), labels, labels)

def get_cv_avg_one_away_accuracy_per_class(p_class):
    get_fold_label = lambda fold, label: get_one_away_accuracy_per_class(p_class[fold][0], p_class[fold][1])[label]
    labels = np.unique(p_class[0][0])
    return {label: np.mean(
        [get_fold_label(fold, label) for fold in range(len(p_class))]) for label in labels}

def get_cv_avg_one_away_accuracy(p_class):
    return np.mean(list(get_cv_avg_one_away_accuracy_per_class(p_class).values()))

def get_one_away_accuracy(y_true, y_pred):
    one_away = lambda p: p[1] in [p[0], p[0]+1, p[0]-1]
    if_one_away_predicted_correct = map(one_away, zip(y_true, y_pred))
    return sum(if_one_away_predicted_correct)/len(y_true)

def get_one_away_accuracy_per_class(y_true, y_pred):
    classes_one_away = {}
    for label in np.unique(y_true):
        label_predictions = [item[1] for item in zip(y_true, y_pred) if item[0]==label]
        classes_one_away[label] = get_one_away_accuracy([label]*len(label_predictions), label_predictions)
    return classes_one_away

def get_cv_multiclass_metrics(p_class, p_proba=None):
    metrics_dict = {}
    n_folds = len(p_class)
    for fold in range(n_folds):    
        fold_metrics_dict = get_multiclass_metrics(p_class[fold], p_proba[fold]) if p_proba else get_multiclass_metrics(p_class[fold])
        for metric in fold_metrics_dict:
            if not metric in metrics_dict:
                metrics_dict[metric] = 0
            metrics_dict[metric] += fold_metrics_dict[metric]/n_folds
    return metrics_dict

def get_multiclass_metrics(p_class, p_proba=None):
    metrics_pred = {
        'one_away_accuracy': get_one_away_accuracy,
        'accuracy_score': metrics.accuracy_score,
        'cohen_kappa_score': metrics.cohen_kappa_score,
        'matthews_corrcoef': metrics.matthews_corrcoef,
        'f1_score': lambda y_true, y_pred: metrics.f1_score(y_true, y_pred, average='weighted'),
        'precision_score': lambda y_true, y_pred: metrics.precision_score(y_true, y_pred, average='weighted'),
        'recall_score': lambda y_true, y_pred: metrics.recall_score(y_true, y_pred, average='weighted'),
    }
    results = {metric: metrics_pred[metric](p_class[0], p_class[1]) for metric in metrics_pred}
    if p_proba:
        results['log_loss'] = metrics.log_loss(p_proba[0], p_proba[1])
        results['roc_auc_score'] = metrics.roc_auc_score(p_proba[0], p_proba[1], multi_class='ovo', average='macro')
    return results