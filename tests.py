import pandas as pd
import numpy as np
import yfinance

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



####
'''
   history = model.fit(X_train, y_train, epochs=20, batch_size=32)

   loss = history.history['loss']

   # Prediction
   y_pred = model.predict(X_test)

   # Compute error metrics
   # MAE
   mae = mean_absolute_error(y_test, y_pred)
   # RMSE
   rmse = mean_squared_error(y_test, y_pred, squared=False)
   # MDA
   mda = np.mean((np.sign(y_test[1:] - y_test[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])).astype(int))

   # Append results
   y_predictions = []
   for i in range(0, len(y_pred)):
       y_predictions.append(float(y_pred[i]))
   y_pred = np.array(y_predictions)
   '''

optimization_results = pd.read_csv('optimization_results.csv', index_col=0)
print(optimization_results.loc[optimization_results['mse'].idxmin()])