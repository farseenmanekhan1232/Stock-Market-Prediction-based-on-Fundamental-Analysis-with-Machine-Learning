import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy

import cvxpy as cp

import os
import glob

from pypfopt.risk_models import semicovariance
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt import risk_models
from pypfopt.risk_models import risk_matrix
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation


def portfolio_optimization(backtesting_data, keep_top_k_stocks):
    unique_tickers = backtesting_data['ticker'].unique().tolist()
    unique_dates = np.sort(backtesting_data['date'].unique())
    tickers_to_delete = []
    for ticker in unique_tickers:
        inside = backtesting_data.loc[backtesting_data['ticker'] == ticker]
        if len(inside['date']) != 6:
            tickers_to_delete.append(ticker)

    for ticker in tickers_to_delete:
        backtesting_data.drop(backtesting_data[backtesting_data['ticker'] == ticker].index, inplace=True)

    unique_tickers = backtesting_data['ticker'].unique().tolist()
    for ticker in unique_tickers:
        backtesting_data['date'].loc[backtesting_data['ticker'] == ticker] = ['2020-06-30', '2020-09-30', '2020-12-31',
                                                                              '2021-03-31', '2021-06-30', '2021-09-30']

    expected_returns = pd.DataFrame()
    for ticker in unique_tickers:
        inside = backtesting_data['expected_returns'].loc[backtesting_data['ticker'] == ticker].values
        expected_returns[ticker] = inside

    expected_returns.index = ['2020-06-30', '2020-09-30', '2020-12-31', '2021-03-31', '2021-06-30', '2021-09-30']

    expected_returns = expected_returns.iloc[:, :keep_top_k_stocks]
    unique_tickers = expected_returns.columns.tolist()

    sigma = calc_covariance(unique_tickers)
    sigma = risk_models.fix_nonpositive_semidefinite(sigma)
    mu = mean_historical_return(expected_returns, returns_data=True, frequency=3)
    ef = EfficientFrontier(mu, sigma)
    ef.add_objective(objective_functions.L2_reg)
    ef.min_volatility()
    weights = ef.clean_weights()

    return weights, unique_tickers


def calc_covariance(unique_tickers):
    df = pd.DataFrame()
    for ticker in unique_tickers:
        inside_df = pd.read_csv(f'Price data/{ticker}.csv')
        inside_df.drop(columns='Date', inplace=True)
        preprocessed_df = preprocessing_historical_prices(inside_df)
        df[ticker] = preprocessed_df
    return risk_models.risk_matrix(df, returns_data=True)


def preprocessing_historical_prices(inside_df):
    inside_df = inside_df.pct_change()
    inside_df.dropna(inplace=True)
    return inside_df


def calc_portfolio_performance(optimal_weights, unique_tickers):
    df = pd.DataFrame()
    for ticker in unique_tickers:
        inside_df = pd.read_csv(f'Price data/{ticker}.csv')
        # filter dates
        inside_df = inside_df.loc[inside_df['Date'].isin(['2020-06-30', '2020-09-30', '2020-12-31', '2021-03-31',
                                                          '2021-06-30', '2021-09-30'])]
        inside_df.drop(columns='Date', inplace=True)
        df[ticker] = inside_df

    # compute weighted returns
    portfolio_value = 10000
    portfolio_components = []
    for i in range(df.shape[0]):
        allocation = DiscreteAllocation(optimal_weights, df.iloc[i, :], total_portfolio_value=portfolio_value, short_ratio=0.3).greedy_portfolio()
        portfolio_value = allocation[1]
        portfolio_components.append(allocation[0])

    allocations = portfolio_components[0]
    for i in range(1, len(portfolio_components) - 1):
        item = portfolio_components[i]
        for key in item.keys():
            allocations[key] = allocations[key] + item[key]


    # calculate portfolio value at the end of backtesting
    last_day_of_backtesting = df.tail(1)
    sum = portfolio_value
    for key, value in allocations.items():
        print(key)
        inside = value * last_day_of_backtesting[key].values[0]
        print(inside)
        sum = sum + inside
    print(sum)


