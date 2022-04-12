import pandas as pd
import numpy as np

backtesting_data = pd.read_csv('backtesting_data.csv', index_col=0)
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
    print(inside)
    expected_returns[ticker] = inside

mu = mean_historical_return
