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
    '''
    # Specify tickers
    tickers = ['IBM', 'AAPL', 'MSFT', 'JNJ', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'FB', 'NVDA', 'BRK-B', 'JPM',
               'UNH', 'HD', 'PG', 'V', 'BAC', 'ADBE', 'MA', 'NFLX', 'CRM', 'PFE', 'XOM', 'TMO', 'CMCSA', 'PYPL',
               'ACN', 'CSCO', 'PEP', 'CVX', 'NKE', 'KO', 'VZ', 'MRK', 'QCOM', 'LLY', 'ABBV', 'WFC', 'WMT', 'INTC',
               'DHR', 'MCD', 'AMD', 'TXN', 'LOW', 'T', 'INTU', 'NEE', 'MDT', 'UNP', 'ORCL', 'HON', 'UPS', 'AMAT', 'PM',
               'MS', 'C', 'NOW', 'SBUX', 'GS', 'BLK', 'BMY', 'RTX', 'ISRG', 'CVS', 'BA', 'TGT', 'SCHW', 'AMT', 'AMGN',
               'AXP', 'SPGI', 'PLD', 'GE', 'CAT', 'ZTS', 'ANTM', 'MMM', 'DE', 'ADI', 'ADP', 'BKNG', 'COP', 'LRCX', 'GM',
               'TJX', 'SYK', 'MDLZ', 'CHTR', 'MU', 'PNC', 'GILD', 'MMC', 'LMT', 'CB', 'TFC', 'CSX', 'CME', 'MO', 'EL',
               'SHW', 'CCI', 'USB', 'F', 'ICE', 'DUK', 'EW', 'BDX', 'EQIX', 'ADSK', 'ETN', 'ITW', 'TMUS', 'COF',
               'HCA', 'ECL', 'BSX', 'JCI', 'FCX', 'HUM', 'EMR', 'ILMN', 'IDXX', 'SPG', 'XLNX', 'SNPS',
               'TEL', 'MSCI', 'NOC', 'DG', 'PGR', 'ROP', 'EXC', 'CDNS', 'INFO', 'IQV', 'APH', 'PSA', 'EOG',
               'A', 'ALGN', 'CMG', 'ATVI', 'TROW', 'APTV', 'EBAY', 'VRTX', 'AIG', 'DLR', 'TT', 'FTNT',
               'BK', 'MCHP', 'GD', 'KMB', 'NEM', 'MET', 'LHX', 'ORLY', 'CTSH', 'MSI', 'SIVB', 'CNC',
               'PH', 'MAR', 'SLB', 'AEP', 'PRU', 'ROK', 'ROST', 'HLT', 'AZO', 'PXD', 'PAYX', 'BAX', 'STZ', 'CTAS',
               'MTCH', 'SRE', 'O', 'TWTR', 'MPC', 'BIIB', 'PPG', 'TRV', 'SYY', 'RMD', 'SBAC', 'HPQ', 'EA', 'GIS',
               'YUM', 'IFF', 'VRSK', 'GPN', 'ADM', 'ENPH', 'KEYS', 'MTD', 'AMP', 'EFX', 'AJG', 'WBA', 'ETSY', 'ALL',
               'MNST', 'WMB', 'AVB', 'CBRE', 'ODFL', 'ALB', 'TDG', 'AME', 'CMI', 'DHI', 'WST', 'ZBRA', 'PEG', 'AWK',
               'PSX', 'KMI', 'BLL', 'BBY', 'PCAR', 'DLTR', 'AME', 'CMI', 'DHI', 'WST', 'ZBRA', 'PEG', 'AWK', 'PSX', 'KMI',
               'BLL', 'BBY', 'PCAR', 'DLTR', 'CPRT', 'FITB', 'SWK', 'LEN', 'WLTW', 'GLW', 'KR', 'WY', 'ES', 'EQR', 'VLO',
               'WEC', 'RSG', 'ANET', 'LUV', 'FTV', 'ARE', 'GNRC', 'OKE', 'SYF', 'LH', 'ZBH', 'ED', 'CDW', 'HSY', 'VMC',
               'TSCO', 'MLM', 'DVN', 'SWKS', 'NTRS', 'DAL', 'DOV', 'KHC', 'EXPE', 'VFC', 'EIX', 'HIG', 'TSN', 'NDAQ',
               'LYB', 'KMX', 'HBAN', 'MPWR', 'VRSN', 'MAA', 'TER', 'CHD', 'KEY', 'ESS', 'XYL', 'POOL', 'HES', 'ULTA',
               'DRE', 'AEE', 'PPL', 'EXPD', 'CERN', 'CTLT', 'GWW', 'DTE', 'CFG', 'TRMB', 'PAYC', 'GRMN', 'ETR', 'MKC',
               'FE',  'TYL', 'WAT', 'MTB', 'CLX', 'HAL', 'TDY', 'STX', 'PKI', 'BR', 'BBWI', 'VIAC', 'GPC', 'CZR', 'COO',
               'DPZ', 'NTAP', 'HPE', 'DRI', 'VTR', 'TTWO', 'HOLX', 'IP', 'J', 'FANG', 'FLT', 'RJF', 'CE', 'CRL', 'WDC',
               'DGX', 'PEAK', 'TECH', 'ABC', 'PFG', 'AVY', 'AKAM', 'WAB', 'CINF', 'IEX', 'NVR', 'CMS', 'RCL',
               'MGM', 'CTRA', 'TXT', 'QRVO', 'MAS', 'JBHT', 'PWR', 'BXP', 'AES', 'LKQ', 'K', 'CCL', 'BIO', 'UDR',
               'EMN', 'CNP', 'AAP', 'BRO', 'CAG', 'UAL', 'ABMD', 'LYV', 'OMC', 'TFX', 'KIM', 'FBHS', 'WHR',
               'SJM', 'LNT', 'NLOK', 'CAH', 'CF', 'FFIV', 'MKTX', 'CBOE', 'LUMN', 'PHM', 'BF-B', 'IRM', 'IPG', 'FMC',
               'LVS', 'RHI', 'MRO', 'PNR', 'CHRW', 'TPR', 'HAS', 'PKG', 'MOS', 'LNC', 'AAL', 'LDOS', 'ATO', 'HST',
               'WRB', 'HWM', 'REG', 'BWA', 'AOS', 'XRAY', 'RE', 'ZION', 'JNPR', 'HSIC', 'ALLE', 'SNA', 'CMA', 'JKHY',
               'L', 'APA', 'NI', 'MHK', 'SEE', 'UHS', 'BEN', 'NRG', 'FRT', 'WYNN', 'NWSA', 'TAP', 'CPB', 'NWL', 'GL', 'DISH',
               'PENN', 'PVH', 'IVZ', 'LW', 'ROL', 'PNW', 'PBCT', 'NCLH', 'DISCK', 'WU', 'DVA', 'VNO', 'ALK', 'IPGP',
               'HBI',  'RL', 'LEG', 'UAA', 'UA', 'DISCA', 'GPS', 'NWS']

    # feature engineering phase
    import featureEngineering_module
    data = featureEngineering_module.featureEngineering(tickers)
    print(data.shape)

    # preprocessing phase
    import preprocessing_module
    X_train, y_train, X_test, y_test, y_train_class, y_test_class, columns, test_data_with_dates = preprocessing_module.preprocessing(data)

    # Run multi-output model
    import regression_models
    reg_pred, class_pred, mae, rmse, acc, rec, prec, f1 = regression_models.NN_MO(X_train, y_train, X_test, y_test,
                                                                                 y_train_class, y_test_class)

    print('Regression metrics')
    print('MAE: ', mae)
    print('RMSE: ', rmse)
    print('Classification metrics')
    print('Accuracy: ', acc)
    print('Recall: ', rec)
    print('Precision: ', prec)
    print('F1 score: ', f1)

    # add date and ticker to predictions
    backtesting_data = test_data_with_dates[['date', 'ticker']]
    backtesting_data.reset_index(inplace=True, drop=True)
    backtesting_data['expected_returns'] = reg_pred
    print(backtesting_data)
    backtesting_data.to_csv('backtesting_data.csv')
    
    '''

    backtesting_data = pd.read_csv('backtesting_data.csv', index_col=0)
    # perform mean-variance portfolio optimization
    keep_top_k_stocks = 10
    optimal_weights, unique_tickers = portfolio_optimization_module.portfolio_optimization(backtesting_data, keep_top_k_stocks)

    # calculate portfolio performance
    portfolio_optimization_module.calc_portfolio_performance(optimal_weights, unique_tickers)

