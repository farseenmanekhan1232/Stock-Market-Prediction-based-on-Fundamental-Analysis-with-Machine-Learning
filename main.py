# Import libraries
import json
import pandas as pd
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    print(tickers)

    # feature engineering phase
    import featureEngineering_module
    data = featureEngineering_module.featureEngineering(tickers)
    print(data.shape)

    # preprocessing phase
    import preprocessing_module
    X_train, y_train, X_test, y_test, y_train_class, y_test_class, columns, test_data_with_dates = preprocessing_module.preprocessing(data)

    # Run multi-output model
    import regression_models
    reg_pred, mae, rmse, xgboostRegressor = regression_models.xgboostRegressor(X_train, y_train, X_test, y_test)

    print('Regression metrics')
    print('MAE: ', mae)
    print('RMSE: ', rmse)

    # Construct new dataset based on xgboost regression results
    X_train, X_test = featureEngineering_module.hybrid_dataset_construction(xgboostRegressor, X_train, X_test)
    columns.append('xgboost_predictions')

    # keep fundamentals and xgboost predictions
    X_train = X_train[:, np.r_[0:10, -1]]
    del columns[13:-1]

    X_test = X_test[:, np.r_[0:10, -1]]

    # feed new data to NN regressor model
    y_pred, mae, rmse, mda, loss = regression_models.NN(X_train, y_train, X_test, y_test)

    # add date and ticker to predictions
    backtesting_data = test_data_with_dates[['date', 'ticker']]
    backtesting_data.reset_index(inplace=True, drop=True)
    backtesting_data['expected_returns'] = reg_pred
    print(backtesting_data)
    backtesting_data.to_csv('backtesting_data.csv')


    #backtesting_data = pd.read_csv('backtesting_data.csv', index_col=0)
    # perform mean-variance portfolio optimization
    keep_top_k_stocks = 5
    optimal_weights, unique_tickers = portfolio_optimization_module.portfolio_optimization(backtesting_data,
                                                                                           keep_top_k_stocks)
    print(optimal_weights)
    # calculate portfolio performance
    portfolio_optimization_module.calc_portfolio_performance(optimal_weights, unique_tickers)

