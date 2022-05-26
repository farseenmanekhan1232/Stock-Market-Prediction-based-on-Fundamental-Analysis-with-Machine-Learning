import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations
from tensorflow.keras.optimizers import Nadam, Adam

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import xgboost


def NN_MO(X_train, y_train, X_test, y_test, y_train_class, y_test_class):
    # input
    visible = tf.keras.layers.Input(shape=(19,))
    hidden1 = tf.keras.layers.Dense(20, activation='relu', kernel_initializer='he_normal')(visible)
    hidden2 = tf.keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal')(hidden1)

    # regression output
    out_reg = tf.keras.layers.Dense(1, activation='linear')(hidden2)

    # classification output
    out_class = tf.keras.layers.Dense(1, activation='sigmoid')(hidden2)

    # define model
    model = tf.keras.Model(inputs=visible, outputs=[out_reg, out_class])

    # compile model
    model.compile(loss=['mse', 'binary_crossentropy'], optimizer='adam')

    # fit model
    # HISTORY PLOTS what does it use
    # LEARNING CURVE
    # SEND ARCHITECTURE PLOT TO LEFTERIS
    model.fit(X_train, [y_train, y_train_class], epochs=100, batch_size=32, verbose=2)

    # make predictions on test set
    yhat1, yhat2 = model.predict(X_test)

    # convert yhat2 to binary
    yhat2_bin = np.array(yhat2)
    for i in range(len(yhat2)):
        if yhat2[i] > 0.5:
            yhat2_bin[i] = 1
        else:
            yhat2_bin[i] = 0

    # regression metrics
    mae = mean_absolute_error(y_test, yhat1)
    rmse = mean_squared_error(y_test, yhat1, squared=False)

    # classification metrics
    accuracy = accuracy_score(y_test_class, yhat2_bin)
    recall = recall_score(y_test_class, yhat2_bin)
    precision = precision_score(y_test_class, yhat2_bin)
    f1score = f1_score(y_test_class, yhat2_bin)

    return yhat1, yhat2_bin, mae, rmse, accuracy, recall, precision, f1score


# Neural Network
def NN(learning_rate):
    # Experiments - layer number, learning, losses
    # Criteria - statistical, financial
    # Optimization criteria in other literature work
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=14, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(64, activation='relu', kernel_initializer='he_normal'))
    model.add(layers.Dense(1, activation='linear'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics='mse')

    # Training
    # define early stopping
    #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    return model


# Random forest regressor
def RFRegressor(X_train, y_train, X_test, y_test):
    # Define dummy regressor model
    regressor = RandomForestRegressor(random_state=42)

    # return optimal hyperparameters
    best_parameters = {'min_samples_split': 2, 'n_estimators': 1000}

    # Declare new regressor with optimal hyperparameters
    regressor = RandomForestRegressor(random_state=42, n_estimators=best_parameters['n_estimators'],
                                      min_samples_split=best_parameters['min_samples_split'])

    # Training
    regressor.fit(X_train, y_train)

    # Prediction
    y_pred = regressor.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    # RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return y_pred, mae, rmse


# Support vector machine regressor
def SVRegressor(X_train, y_train, X_test, y_test):
    # Define dummy regressor model
    regressor = SVR()

    # return optimal hyperparameters
    best_parameters = {'C': 1, 'kernel': 'rbf'}

    # Declare new regressor with optimal hyperparameters
    regressor = SVR(kernel=best_parameters['kernel'], C=best_parameters['C'])

    # Training
    regressor.fit(X_train, y_train)

    # Prediction
    y_pred = regressor.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    # RMSE
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return y_pred, mae, rmse


def xgboostRegressor(X_train, y_train, X_test, y_test):
    # Define dummy regressor model
    regressor = xgboost.XGBRegressor(random_state=42)

    # return optimal hyperparameters
    best_parameters = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}

    # Declare new regressor with optimal hyperparameters
    regressor = xgboost.XGBRegressor(random_state=42, n_estimators=best_parameters['n_estimators'],
                                     max_depth=best_parameters['max_depth'],
                                     learning_rate=best_parameters['learning_rate'])

    # Training
    regressor.fit(X_train, y_train)

    # Prediction
    y_pred = regressor.predict(X_test)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)

    # MSE
    mse = mean_squared_error(y_test, y_pred)

    # R2
    r2 = r2_score(y_test, y_pred)

    return y_pred, mae, mse, r2, regressor
