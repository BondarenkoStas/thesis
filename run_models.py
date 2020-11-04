import time
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
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

from importlib import reload
import output, process_df
from output import output_metrics, mape, smape, get_metrics, print_sorted_actual_to_predicted_graphs_only_test
from process_df import Process, create_average_columns, split_process_df, split_df
reload(output)
reload(process_df)
from output import output_metrics, mape, smape, get_metrics, print_sorted_actual_to_predicted_graphs_only_test
from process_df import Process, create_average_columns, split_process_df, split_df

def get_df_work_columns(df, df_columns):
    return df[[col for col in df_columns if not 'META' in col or col == 'META__revenue']]

def create_and_fit_regression_rf(X, y, X_val=None, y_val=None, verbose=0):
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
        verbose=verbose,
    )
    model.fit(X, y)
    return model


def create_regression_lgb():
    return lgb.LGBMRegressor(
        objective='huber',
        num_leaves=34,
        learning_rate=0.01, 
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

def create_and_fit_regression_lgb(X, y, X_val, y_val, verbose=0, patience=30):
    model = create_regression_lbg()
    model = model.fit(X.values, y,
        verbose=verbose,
        eval_set=(X_val, y_val),
        early_stopping_rounds=patience
    )
    return model

def build_nn_model(input_shape=None, loss=MeanSquaredError()):
    def get_model():
        adamax = Adamax(learning_rate=0.001,beta_1=0.958,beta_2=0.987)
        model = Sequential([
            Dense(
                256, 
                activation='sigmoid', 
                input_shape=input_shape,
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
            Dense(1, kernel_initializer='glorot_normal')
        ])

        model.compile(loss=loss,
                    optimizer=adamax,
                    metrics=['mae', 'mse'])
        return model
    return get_model

def create_and_fit_regression_nn(X, y, X_val, y_val, loss=MeanSquaredError(), verbose=0, patience=5, regressor_params={}):
    model = build_nn_model([X.shape[1]], loss=loss)()
    es = EarlyStopping(
        monitor='val_loss',
        mode='min', 
        verbose=verbose, 
        patience=patience)

    history = model.fit(X, y,
        epochs=10000, 
        validation_data=(X_val, y_val),
        verbose=verbose,
        batch_size=256,
        shuffle=True,
        callbacks=[es])
    return model, history

class NN_estimator(BaseEstimator, RegressorMixin):
    def __init__(self, print_graphs = False):
        self.print_graphs = print_graphs

    
    def fit(self, X, y, **kwargs):
        self.process_ = kwargs['process']
        model = build_nn_model([data['X_train'].shape[1]])

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


def get_classification_cv_predictions(
    create_and_fit_fn, 
    X, 
    y, 
    n_splits=10, 
    validation_size=0.05, 
    patience=100, 
    model_params={}, 
    verbose=0, 
    validation_split=None,
    convert_y_categorical=False,
):
    start = time.time()
    predicted_proba = []
    predicted_class = []
    history = []

    split_num=0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_index, test_index in skf.split(X, y):
        split_num+=1
        print(f'split num: {split_num}')
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        if convert_y_categorical:
            y_train, y_test = to_categorical(y_train), to_categorical(y_test)
        if validation_split:
            X_val, y_val = None, None
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)
        model_return = create_and_fit_fn(X_train, y_train, X_val, y_val,
            patience=patience, model_params=model_params.copy(), verbose=verbose, validation_split=validation_split)

        if not isinstance(model_return, tuple):
            model_return = (model_return, None)
        if convert_y_categorical:
            y_test = np.argmax(y_test, axis=-1)
        predicted_proba.append((y_test, model_return[0].predict_proba(X_test)))
        predicted_class.append((y_test, model_return[0].predict(X_test)))
        history.append(model_return[1])
    end = time.time()
    print(f'{n_splits} CV time: {end - start}')
    return predicted_proba, predicted_class, history

def get_regression_cv_metrics(create_and_fit_fn, X, y, n_splits=10, patience=30, validation_size=0.05, 
                                model_params={}, should_output_metrics=False, should_output_graphs=False, verbose=0, validation_split=None):
    from sklearn.model_selection import KFold, train_test_split
    start = time.time()
    cv_results = []
    history = []
    split_num=0
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(X, y):
        split_num+=1
        print(f'split {split_num}')
        X_train_and_val, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train_and_val, y_test = y[train_index], y[test_index]
        if validation_split:
            model_return = create_and_fit_fn(X_train_and_val, y_train_and_val, patience=patience, regressor_params=model_params.copy(), verbose=verbose, validation_split=validation_split)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=validation_size)
            model_return = create_and_fit_fn(X_train, y_train, X_val, y_val, patience=patience, regressor_params=model_params.copy(), verbose=verbose)
        if not isinstance(model_return, tuple):
            model_return = (model_return, None)
        cv_results.append(output_metrics(model_return[0], X, y, X_test, y_test, 
            should_output_graphs=should_output_graphs, should_output_metrics=should_output_metrics, should_return_result=False))
        history.append(model_return[1])
    metrics = ['smape', 'mape', 'mae', 'rmse', 'adj_r2']
    metric_results = {metric: np.mean([cv_results[split_index]['test'][metric] for split_index in range(n_splits)]) for metric in metrics}
    print(metric_results)
    end = time.time()
    print(f'{n_splits} CV time: {end - start}')
    return {'cv_iterations': cv_results, 'cv_metrics': metric_results, 'history': history}

def get_regression_cv_predictions(create_and_fit_fn, X, y, n_splits=10, patience=30, validation_size=0.05):
    test_true = []
    test_pred = []

    split_num=0
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(X, y):
        split_num+=1
        print(f'split {split_num}')
        X_train_and_val, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train_and_val, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=validation_size)

        test_true.extend(y_test)
        test_pred.extend(create_and_fit_fn(X_train, y_train, X_val, y_val, patience=patience).predict(X_test))
    return test_true, test_pred