import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.compat.v1.keras import backend as K

def keras_conf_matrix(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return tp, tn, fp, fn

def keras_error_rate(y_true, y_pred):
    tp, tn, fp, fn = keras_conf_matrix(y_true, y_pred)
    return (fp + fn) / (tp + tn + fp + fn)

def keras_matthews_correlation(y_true, y_pred):
    tp, tn, fp, fn = keras_conf_matrix(y_true, y_pred)
    return (tp * tn - fp * fn)/(K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + K.epsilon())

def keras_f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def calc_binary_metrics(y_true, y_pred):
    metric_fn = {
        'accuracy': metrics.accuracy_score,
        'f1': metrics.f1_score,
        'roc_auc': metrics.roc_auc_score,
        'cohen_kappa': metrics.cohen_kappa_score,
        'matthews': metrics.matthews_corrcoef,
        'precision': metrics.precision_score,
        'recall': metrics.recall_score
    }
    return {metric: metric_fn[metric](y_true, y_pred) for metric in metric_fn}

def calc_binary_metrics_all_threshold(y_true, y_pred):
    threshold_array = []
    for i in np.linspace(0, 1, 101):
        y_pred_class = y_pred[:, 1] > i
        threshold_array.append(calc_binary_metrics(y_true, y_pred_class))
    return {metric: [threshold_array[i][metric] for i in range(len(threshold_array))] for metric in threshold_array[0]}

def calc_confusion_values(y_true, y_pred):
    conf_array = {'tn': [], 'fp': [], 'fn': [], 'tp': [], 'error_rate':[]}

    for i in np.linspace(0, 1, 101):
        y_pred_class = y_pred[:, 1] > i
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_class).flatten()
        conf_array['tn'].append(tn)
        conf_array['fp'].append(fp)
        conf_array['fn'].append(fn)
        conf_array['tp'].append(tp)
        conf_array['error_rate'].append(fp + fn)
    return conf_array

def get_binary_metrics_for_threshold(metrics_array, threshold):
    for k in metrics_array:
        print(f'{k}: {metrics_array[k][int(threshold*100)]}')

def print_binary_metrics(metric_array, print_graph=True):
    for metric in metric_array:
        m_max = max(metric_array[metric])
        m_min = min(metric_array[metric])
        print(f'{metric}: max {m_max} at threshold {metric_array[metric].index(m_max)/100}')
        print(f'{metric}: min {m_min} at threshold {metric_array[metric].index(m_min)/100}')
        if print_graph:
            plt.plot(np.linspace(0, 1, len(metric_array[metric])), metric_array[metric])
    if print_graph:
        plt.xlabel('threshold')
        plt.legend(list(metric_array.keys()))
        plt.show()

def get_print_binary_metrics(y_true, y_pred):
    binary_metrics = calc_binary_metrics(y_true, y_pred)
    confusion_values = calc_confusion_values(y_true, y_pred)
    print_binary_metrics(binary_metrics)
    print_binary_metrics(confusion_values)
    return {**binary_metrics, **confusion_values}

def get_binary_cv_metrics(predictions, print_graphs=True):
    cv_metrics = []
    for split in predictions:
        y_true, y_pred = split
        # res = calc_binary_metrics(y_true, y_pred)
        res = calc_binary_metrics_all_threshold(y_true, y_pred)
        res.update(calc_confusion_values(y_true, y_pred))
        cv_metrics.append(res)

    avg_for_each_threshold = {}

    n_folds = len(cv_metrics)
    metric_keys = list(cv_metrics[0].keys())
    n_thresholds = len(cv_metrics[0][metric_keys[0]])

    for metric in metric_keys:
        threshold_values = []
        for threshold in range(n_thresholds):
            threshold_values.append(np.mean([cv_metrics[fold][metric][threshold] for fold in range(n_folds)]))
        avg_for_each_threshold[metric] = threshold_values

    if print_graphs:
        metrics_conf = ['tp', 'tn', 'fp', 'fn', 'error_rate']
        metrics_norm = [k for k in avg_for_each_threshold.keys() if k not in metrics_conf]

        print_binary_metrics({m:avg_for_each_threshold[m] for m in metrics_norm})
        print_binary_metrics({m:avg_for_each_threshold[m] for m in metrics_conf})

    return avg_for_each_threshold