# Import libraries
import json
import pandas as pd
import math
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.losses import Huber, CosineSimilarity, MeanSquaredLogarithmicError

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

import xgboost

from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
import yfinance as yf

import portfolio_optimization_module

if __name__ == '__main__':
    # Specify tickers
    tickers = open('tickers.txt', 'r')
    tickers = tickers.read().split(",")

    # feature engineering phase
    import featureEngineering_module
    data = featureEngineering_module.featureEngineering(tickers)
    print(data.shape)

    # preprocessing phase
    import preprocessing_module
    X_train, y_train, X_test, y_test, y_train_class, y_test_class, columns,\
    test_data_with_dates, X_val, y_val, y_val_class, validation_data_with_dates = preprocessing_module.preprocessing(data)

    # run xgboost regressor model
    import regression_models
    reg_pred, mae, mse, r2, xgboostRegressor = regression_models.xgboostRegressor(X_train, y_train, X_test, y_test)

    # report regression metrics
    print('Regression metrics')
    print('MAE: ', mae)
    print('MSE: ', mse)
    print('R2: ', r2)

    # Construct new dataset based on xgboost regression results
    X_train, X_test, X_val = featureEngineering_module.hybrid_dataset_construction(xgboostRegressor, X_train, X_test, X_val)
    columns.append('xgboost_predictions')

    # keep fundamentals and xgboost predictions
    X_train = X_train[:, np.r_[0:13, -1]]
    X_val = X_val[:, np.r_[0:13, -1]]
    del columns[13:-1]

    X_test = X_test[:, np.r_[0:13, -1]]

    '''
    # perform grid search on validation set
    # define the grid search parameters
    grid = preprocessing_module.grid_construction()
    hyperparameter_list = []
    mse_list = []
    mae_list = []
    r2_list = []
    portfolio_value_list = []
    weights_list = []
    port_volatility = []
    y_hat_val_list = []
    for batch_size in grid['batch_size']:
        for epoch in grid['epochs']:
            for loss in grid['loss']:
                model = regression_models.NN(learning_rate=0.01, loss=loss)
                model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch)
                y_hat_val = model.predict(X_val)
                y_hat_val_list.append(y_hat_val)
                mse, mae, r2 = preprocessing_module.evaluation(y_val, y_hat_val)
                hyperparameter_list.append(f'batch_size:{batch_size}, epochs:{epoch}, loss:{loss}')
                mse_list.append(mse)
                mae_list.append(mae)
                r2_list.append(r2)

                # add date and ticker to predictions
                backtesting_data = validation_data_with_dates[['date', 'ticker']]
                backtesting_data.reset_index(inplace=True, drop=True)
                backtesting_data['expected_returns'] = y_hat_val

                # portfolio optimization
                keep_top_k_stocks = 10
                optimal_weights, unique_tickers = portfolio_optimization_module.portfolio_optimization(backtesting_data,
                                                                                                       keep_top_k_stocks)
                weights_list.append(optimal_weights)
                # calculate portfolio performance
                portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(
                    validation_data_with_dates, y_hat_val)

                portfolio_value_list.append(cum_ret.iloc[-1].values)


    
    optimization_results = pd.DataFrame([hyperparameter_list, mse_list, mae_list, r2_list, portfolio_value_list],
                                        index=['hyperparameters', 'mse', 'mae', 'r2', 'cum return'])
    optimization_results = optimization_results.T

    optimization_results.index = optimization_results['hyperparameters']
    optimization_results.drop(columns='hyperparameters', inplace=True)

    optimization_results = optimization_results.astype(float).round(4)

    optimization_results.to_csv('optimization_results.csv')
    '''
    optimization_results = pd.read_csv('optimization_results.csv', index_col=0)

    # select model that minimizes mse
    # AFTER EXPERIMENTATION: batch_size:60, epochs:20, loss:Huber
    print(optimization_results.loc[optimization_results['mse'].idxmin()])

    # select model that minimizes mae
    # AFTER EXPERIMENTATION: batch_size:60, epochs:20, loss:Huber
    print(optimization_results.loc[optimization_results['mae'].idxmin()])

    # select model tha maximizes R-squared
    # AFTER EXPERIMENTATION: batch_size:60, epochs:20, loss:Huber
    print(optimization_results['r2'].idxmax())

    # select model that maximizes portfolio profitability
    # AFTER EXPERIMENTATION: batch_size:100, epochs:30, learning_rate:CosineSimilarity
    print(optimization_results.loc[optimization_results['cum return'].idxmax()])

    # train and test model #1 (min mse)
    # reshape y_val
    y_val = y_val.reshape(y_val.shape[0], 1)
    model = regression_models.NN(learning_rate=0.01, loss=Huber())
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=60, epochs=20, validation_data=(X_val, y_val), verbose=False)

    plt.figure(1)
    plt.plot(history.history['loss'], color='b', label='train loss')
    plt.plot(history.history['val_loss'], color='r', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Huber loss')
    plt.legend()
    plt.title('Training loss for MIN MSE model')


    # store predictions on a list
    predictions = []

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    mse, mae, r2 = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

    # train and test model #2 (min mae)
    # reshape y_val
    #y_val = y_val.reshape(y_val.shape[0], 1)
    model = regression_models.NN(learning_rate=0.01, loss=Huber())
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=60, epochs=20, validation_data=(X_val, y_val), verbose=False)

    plt.figure(2)
    plt.plot(history.history['loss'], color='b', label='train loss')
    plt.plot(history.history['val_loss'], color='r', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Huber loss')
    plt.legend()
    plt.title('Training loss for MIN MAE model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    mse, mae, r2 = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

    # train and test model #3 (max R-squared)
    # reshape y_val
    model = regression_models.NN(learning_rate=0.01, loss=Huber())
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=60, epochs=20, validation_data=(X_val, y_val), verbose=False)

    plt.figure(3)
    plt.plot(history.history['loss'], color='b', label='train loss')
    plt.plot(history.history['val_loss'], color='r', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Huber loss')
    plt.legend()
    plt.title('Training loss for MAX R-Squared model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    predictions.append(y_pred)
    mse, mae, r2 = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates,
                                                                                                y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)

    # train and test model #4 (max portfolio return)
    # reshape y_val
    model = regression_models.NN(learning_rate=0.01, loss=CosineSimilarity())
    history = model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0),
                        batch_size=100, epochs=30, validation_data=(X_val, y_val), verbose=False)

    plt.figure(4)
    plt.plot(history.history['loss'], color='b', label='train loss')
    plt.plot(history.history['val_loss'], color='r', label='val_loss')
    plt.xlabel('epochs')
    plt.ylabel('Cosine similarity loss')
    plt.legend()
    plt.title('Training loss for MAX PROFIT model')

    # make predictions on test set and evaluate
    y_pred = model.predict(X_test)
    predictions.append(y_pred)
    mse, mae, r2 = preprocessing_module.evaluation(y_test, y_pred)
    print('MSE =', np.round(mse, 5))
    print('MAE =', np.round(mae, 5))
    print('R2 =', np.round(r2, 5))

    # financial evaluation
    portfolio_ret, cum_ret, sharpe_ratio, volatility = preprocessing_module.financialEvaluation(test_data_with_dates, y_pred)
    print('Cumulative return =', cum_ret.iloc[-1].values)
    print('Sharpe ratio', sharpe_ratio.values)
    print('Volatility', volatility.values)
    print('Detailed portfolio returns', portfolio_ret)


    # scatter plot predictions vs real
    plt.figure(5)
    plt.scatter(test_data_with_dates['date'].values, predictions[0], color='b', label='min mse model', s=25)
    plt.scatter(test_data_with_dates['date'].values, predictions[1], color='g', label='min mae model', s=25)
    plt.scatter(test_data_with_dates['date'].values, predictions[2], color='g', label='max R-squared model', s=25)
    plt.scatter(test_data_with_dates['date'].values, predictions[3], color='c', label='max return model', s=25)
    plt.scatter(test_data_with_dates['date'].values, y_test, color='r', label='real values', s=25, alpha=0.5)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.legend()
    plt.title('Correlation between predictions and real returns')


