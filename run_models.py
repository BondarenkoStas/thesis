import lightgbm as lgb
import tensorflow
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

from importlib import reload
import output
from output import output_metrics
reload(output)
from output import output_metrics


def run_lgb(data, process):
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
    return output_metrics(model, data, process, with_val=True), model


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

    model.compile(loss='mse',
                optimizer=adamax,
                metrics=['mae', 'mse'])
    return model


def run_nn(data, process, with_val=True):
    input_shape = len(data['X_train'].keys())
    model = build_nn_model(input_shape)

    es = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=20)

    model.fit(
        data['X_train'], data['y_train'],
        epochs=10000, 
        validation_data=(data['X_test'], data['y_test']),
        verbose=0,
        batch_size=256,
        shuffle=True,
        callbacks=[es])
    print('-----------------------------------------------')
    print('output NN')
    return output_metrics(model, data, process, with_val), model