import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from sklearn import metrics

pd.options.display.float_format = '{:20,.15f}'.format


def smape(A,P):
    return 100/len(A) * np.sum(2 * np.abs(P - A) / (np.abs(A) + np.abs(P)))

def mape(A, P):
    return 100 * np.mean(np.abs((A - P)/A))

def get_metrics(y_test, y_pred, cols, name):
    SS_Residual = sum((y_test - y_pred)**2)
    SS_Total = sum((y_test - np.mean(y_test))**2)
    r2 = 1 - (float(SS_Residual))/SS_Total
    adj_r2 = 1 - (1-r2)*(len(y_test) - 1)/(len(y_test) - cols -1)
    return  {
        'smape': smape(y_test, y_pred),
        'mape': mape(y_test, y_pred),
        'mae': np.rint(metrics.mean_absolute_error(y_test, y_pred)),
        'rmse': np.rint(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
        'adj_r2': adj_r2,
    }

def print_metrics(y_test, y_pred, cols, name):
    print(name)
    res = get_metrics(y_test, y_pred, cols, name)
    for key in res:
        print(f'{key}: {res[key]:,.3f}')
    return res


def predict(model, X_train, y_train, X_test, y_test, print_res=True):
    num_col = X_train.shape[1]
    func_metrics = print_metrics if print_res else get_metrics

    pred_train = model.predict(X_train.values)
    pred_test = model.predict(X_test.values)
    if len(pred_train.shape) == 2 and pred_train.shape[1] == 1:
        pred_train = pred_train.flatten()
        pred_test = pred_test.flatten()
    return {
        'res': {
            'train': func_metrics(y_train, pred_train, num_col, 'train'),
            'test': func_metrics(y_test, pred_test, num_col, 'test'),
        },
        'predictions':{
            'train': pred_train,
            'test': pred_test,
        },
        'actual': {
            'train': y_train,
            'test': y_test,
        },
    }


def output_metrics(model, data, process, with_val):
    num_cols = data['X_train'].shape[1]
    data_input = [data['X_train'], data['y_train']]
    data_input.extend(
        [data['X_val'], data['y_val']] if with_val else [data['X_test'], data['y_test']])
    res = predict(model, *data_input, print_res=False)

    res_back = [
        res['actual']['train'],
        res['predictions']['train'],
        res['actual']['test'],
        res['predictions']['test'],
    ]

    for apply_function in reversed(process.y_process):
        res_back = [apply_function(arr) for arr in res_back]
    res_back = [np.rint(arr) for arr in res_back]

    res_train = print_metrics(*res_back[:2], num_cols, 'train')
    res_test = print_metrics(*res_back[2:], num_cols, 'test')
    print_sorted_actual_to_predicted_graphs(*res_back, print_log=True)
    return {'train': res_train, 'test': res_test, 'result': res_back}


def print_graphs(actual, predicted, title, print_log=False):
    length = len(actual)
    res_df = pd.DataFrame({'actual':actual, 'predicted': predicted}).astype(float)
    res_df.sort_values(by='actual', inplace=True)

    plt.figure()
    x = np.linspace(0, length, length)
    plt.plot(x, res_df['actual'], label='revenue actual')
    plt.plot(x, res_df['predicted'], label='revenue predicted')

    plt.ticklabel_format(useOffset=False, style='plain')
    plt.rcParams["figure.figsize"] = (8,8)
    if print_log:
        plt.yscale('log')
    plt.xlabel('movie')
    plt.ylabel('revenue')
    plt.title(title)
    plt.legend()
    plt.show()
    
def print_sorted_actual_to_predicted_graphs(train_real, train_pred, test_real, test_pred, print_log=False):
    if print_log:
        print_graphs(train_real, train_pred, 'train data: log scale', print_log=True)
        print_graphs(test_real, test_pred, 'validation data: log scale', print_log=True)
    print_graphs(train_real, train_pred, 'train data')
    print_graphs(test_real, test_pred, 'validation data')