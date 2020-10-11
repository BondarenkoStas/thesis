import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold

from importlib import reload
import output, process_df
from output import output_metrics, mape, smape, get_metrics, print_sorted_actual_to_predicted_graphs_only_test
from process_df import Process, create_average_columns, split_process_df, split_df
reload(output)
reload(process_df)
from output import output_metrics, mape, smape, get_metrics, print_sorted_actual_to_predicted_graphs_only_test
from process_df import Process, create_average_columns, split_process_df, split_df

def get_df_work_columns(df):
    return df[[col for col in df_full.columns if not 'META' in col or col == 'META__revenue']]

def run_rf(data, process, with_val=True):
    model = RandomForestRegressor(
        n_estimators=40,
        max_depth=15,
        min_samples_split=0.001,
        min_samples_leaf=0.0005,
        bootstrap=True,
        max_samples=0.95,
        criterion='mae', 
        random_state=0, 
        n_jobs=-1,
    )
    model.fit(data['X_train'], data['y_train'])
    return output_metrics(model, data, process, with_val), model


def run_lgb(data, process, with_val=True):
    model = lgb.LGBMRegressor(
        objective='huber',
        num_leaves=34,
        learning_rate=0.001, 
        n_estimators=7500,
        max_bin=192,
        max_depth=0,
        min_child_samples=160,
        min_child_weight=0.001,
        bagging_fraction=0.98,
        bagging_freq=15, 
        feature_fraction=0.77,
        bagging_seed=9,
        min_data_in_leaf=1, 
        min_sum_hessian_in_leaf=50,
        colsample_bytree=0.87,
        reg_alpha=0.18,
        reg_lambda=30,
        subsample=0.39,
        tree_learner='data',
    )
    model.fit(
        data['X_train'].values, 
        data['y_train'],
        verbose=0,
#         eval_metric=eval_metric,
        eval_set=[(data['X_test'], data['y_test'])],
        early_stopping_rounds=100
    )
    print('-----------------------------------------------')
    print('output LGBMRegressor')
    return output_metrics(model, data, process, with_val), model


def build_nn_model(input_shape):
    adamax = Adamax(learning_rate=0.001,beta_1=0.958,beta_2=0.987)
    model = Sequential([
    Dense(
        256, 
        activation='sigmoid', 
        input_shape=[input_shape],
        kernel_initializer='glorot_normal',
        kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001),
        bias_regularizer=l1_l2(l1=0.001, l2=0.1)
    ),
    Dropout(0.005),
    Dense(
        256, 
        activation='sigmoid',
        kernel_initializer='glorot_normal',
        kernel_regularizer=l1_l2(l1=0, l2=0.001),
        bias_regularizer=l1_l2(l1=0.01, l2=0.01),
    ),
    Dropout(0.5),  
    Dense(
        1,
        kernel_initializer='glorot_normal',
        activation='linear'
    )
    ])

    model.compile(loss=MeanSquaredError(),
                optimizer=adamax,
                metrics=['mae', 'mse'])
    return model


def run_nn(data, process, with_val=True, verbose=0):
    input_shape = len(data['X_train'].keys())
    model = build_nn_model(input_shape)

    es = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=50)

    model.fit(
        data['X_train'], data['y_train'],
        epochs=10000, 
        validation_data=(data['X_test'], data['y_test']),
        verbose=verbose,
        batch_size=256,
        shuffle=True,
        callbacks=[es])
    print('-----------------------------------------------')
    print('output NN')
    return output_metrics(model, data, process, with_val)


class NN_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, print_graphs = False):
        self.print_graphs = print_graphs

    
    def fit(self, X, y, **kwargs):
        self.process_ = kwargs['process']
        input_shape = len(data['X_train'].keys())
        model = build_nn_model(input_shape)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

        model.fit(X, y, epochs=10000, validation_split=0.1, verbose=0, batch_size=256,
            shuffle=True, callbacks=[es], use_multiprocessing=True)
        self.model_ = model
        return self
    
    def score(self, X, y):
        yhat = self.model_.predict(X.values)
        if len(yhat.shape) == 2 and yhat.shape[1] == 1:
            yhat = yhat.flatten()

        print_log=False
        for apply_function in reversed(self.process_.y_process):
            print_log=True
            y = apply_function(y)
            yhat = apply_function(yhat)

        if self.print_graphs:
            print_sorted_actual_to_predicted_graphs_only_test(y, yhat, print_log=print_log)
        return get_metrics(y, yhat, X.shape[1])


class OutliersEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, outliers_detector=None, drop_index_list=None, folds=10, print_graphs=False):
        self.outliers_detector = outliers_detector
        self.drop_index_list = drop_index_list
        self.folds = folds
        self.print_graphs = print_graphs

    def fit(self, X, y, **kwargs):
        df_raw = pd.read_csv('outliers/df_raw.csv', index_col='id')

        print("######################################################")
        if self.outliers_detector is not None:
            print(self.outliers_detector.get_params())
            X_full = X.copy()
            X_full['revenue'] = y
            mask_to_drop = self.outliers_detector.fit_predict(X_full) == -1
            drop_index_list = df_raw[mask_to_drop].index
        else:
            drop_index_list = self.drop_index_list
        self.drop_index_list_ = drop_index_list
        print('Number of outliers: ', len(drop_index_list))
        new_df = df_raw.drop(drop_index_list)
        df = create_average_columns(new_df, verbose=0)
        print('Average created')
        df = get_df_work_columns(df)
        data, self.process_ = split_process_df(df)
        print('Processed')
        self.X_ = pd.concat([data['X_train'], data['X_test'], data['X_val']])
        self.y_ = np.concatenate([data['y_train'], data['y_test'], data['y_val']])
        return self

    def score(self, X=None, y=None):
        cv_results = []
        kf = KFold(n_splits=self.folds, random_state=0)
        for train_index, test_index in kf.split(self.X_, self.y_):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            estimator = NN_estimator(print_graphs=self.print_graphs)
            estimator.fit(X_train, y_train, process=self.process_)
            fm = estimator.score(X_test, y_test)
            print(f'smape: {fm["smape"]}, mape: {fm["mape"]}, mae: {fm["mae"]}')
            cv_results.append(fm)
        metrics = list(cv_results[0].keys())
        metric_results = {metric: np.mean([cv_results[split_index][metric] for split_index in range(len(cv_results))]) for metric in metrics}
        print(metric_results)
        return {'cv_iterations': cv_results, 'cv_metrics': metric_results}