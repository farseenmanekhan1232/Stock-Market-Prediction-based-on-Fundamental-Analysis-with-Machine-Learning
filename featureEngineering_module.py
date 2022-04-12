import json
import pandas as pd
import numpy as np


def featureEngineering(tickers):
    # Import fundamental and price data and construct features
    data = pd.DataFrame()
    for ticker in tickers:
        # Import fundamental data
        with open(f'data/{ticker}_income_statement.json') as json_file:
            income_statement = json.load(json_file)
        with open(f'data/{ticker}_balance_sheet.json') as json_file:
            balance_sheet = json.load(json_file)
        with open(f'data/{ticker}_cash_flow.json') as json_file:
            cash_flow = json.load(json_file)

        # Import price data
        prices = pd.read_csv(f'data/prices/{ticker}_quarters.csv')
        prices['ticker'] = ticker

        # FEATURE 1 - Return on Assets(ROA) = Net Income / Total Assets
        # -------------------------------------------------------------
        quarterlyIS = income_statement['quarterlyReports']
        netIncome = []
        dates = []
        for quarter in quarterlyIS:
            netIncome.append(int(quarter['netIncome']))
            dates.append(quarter['fiscalDateEnding'])

        ROA = pd.DataFrame(netIncome, columns=['netIncome'])

        # Select total assets from balance sheet
        quarterlyBS = balance_sheet['quarterlyReports']
        totalAssets = []
        for quarter in quarterlyBS:
            if quarter['totalAssets'] == 'None':
                totalAssets.append(0)
            else:
                totalAssets.append(int(quarter['totalAssets']))

        ROA['totalAssets'] = totalAssets

        # Compute return on assets
        ROA['ReturnOnAssets'] = ROA['netIncome'] / ROA['totalAssets']

        # FEATURE 2 - Debt Ratio = Total Liabilities / Total Assets
        # -------------------------------------------------------------
        totalLiabilities = []
        for quarter in quarterlyBS:
            if quarter['totalLiabilities'] == 'None':
                totalLiabilities.append(0)
            else:
                totalLiabilities.append(int(quarter['totalLiabilities']))

        DR = pd.DataFrame()
        DR['totalLiabilities'] = totalLiabilities
        DR['totalAssets'] = totalAssets

        # Compute Debt Ratio
        DR['DebtRatio'] = DR['totalLiabilities'] / DR['totalAssets']

        # FEATURE 3 - Current Ratio = Current Debt / Current Assets
        # -------------------------------------------------------------
        currentDebt = []
        for quarter in quarterlyBS:
            if quarter['currentDebt'] == 'None':
                currentDebt.append(0)
            else:
                currentDebt.append(int(quarter['currentDebt']))

        currentAssets = []
        for quarter in quarterlyBS:
            if quarter['totalCurrentAssets'] == 'None':
                currentAssets.append(0)
            else:
                currentAssets.append(int(quarter['totalCurrentAssets']))

        CR = pd.DataFrame()
        CR['currentDebt'] = currentDebt
        CR['currentAssets'] = currentAssets

        # Compute Current Ratio
        CR['currentRatio'] = CR['currentDebt'] / CR['currentAssets']

        # FEATURE 4 - Gross Margin = Revenue - Cost of Goods sold / Revenue
        # -------------------------------------------------------------
        revenue = []
        for quarter in quarterlyIS:
            if quarter['totalRevenue'] == 'None':
                revenue.append(0)
            else:
                revenue.append(int(quarter['totalRevenue']))

        costOfGoodsSold = []
        for quarter in quarterlyIS:
            if quarter['costofGoodsAndServicesSold'] == 'None':
                costOfGoodsSold.append(0)
            else:
                costOfGoodsSold.append(int(quarter['costofGoodsAndServicesSold']))

        GM = pd.DataFrame()
        GM['revenue'] = revenue
        GM['costOfGoodsSold'] = costOfGoodsSold

        # Add fundamental data
        # Revenue
        FD = pd.DataFrame()
        FD['revenue'] = revenue

        # Expenses
        expenses = []
        for quarter in quarterlyIS:
            expenses.append(quarter['operatingExpenses'])
        FD['expenses'] = expenses

        # Interests
        interests = []
        for quarter in quarterlyIS:
            interests.append(quarter['interestIncome'])
        FD['interests'] = interests
        FD['interests'].replace('None', 0, inplace=True)

        # Net income
        FD['net_income'] = netIncome

        # Gross Profit
        FD['gross_profit'] = FD['revenue'] - costOfGoodsSold

        # Investments
        quarterlyCF = cash_flow['quarterlyReports']
        investments = []
        for quarter in quarterlyCF:
            investments.append(quarter['cashflowFromInvestment'])
        FD['investments'] = investments
        # Liabilities
        FD['liabilities'] = totalLiabilities
        # Assets
        FD['assets'] = totalAssets
        # Equity
        FD['equity'] = FD['assets'] - FD['liabilities']
        # Debt
        FD['debt'] = currentDebt
        # Ticker
        FD['ticker'] = ticker
        # Date
        FD['date'] = dates

        # Merge with fundamental ratios
        FD['return on assets'] = ROA['ReturnOnAssets']
        FD['debt ratio'] = DR['DebtRatio']
        FD['current ratio'] = CR['currentRatio']

        # Remove Open, Low, Adj Close, Volume for prices dataframe
        prices = prices.drop(columns=['Open', 'High', 'Low', 'Adj Close'])
        # prices = prices.drop(columns=['Adj Close'])
        # Reindex prices so that old values are on top
        prices = prices.reindex([18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        prices.reset_index(drop=True, inplace=True)

        # Compute simple returns
        prices['simple returns'] = prices['Close'].pct_change()

        # Compute log returns
        prices['log returns'] = np.log(prices['Close']) - np.log(prices['Close'].shift(1))

        # Compute log returns trend
        prices['log returns trend'] = prices['log returns'].apply(lambda x: 1 if x > 0 else 0)
        # Import expected eps data
        try:
            expected_eps = pd.read_csv(f'data/{ticker}_expected_eps.csv')
            FD = FD.merge(expected_eps, how='outer', on='ticker')
        except FileNotFoundError:
            print('No such file or directory')
        # Merge Fundamental and price data
        FD.sort_values(by=['date'], inplace=True)
        FD = FD.merge(prices, how='outer', on=['date', 'ticker'])
        data = data.append(FD)

    return data
