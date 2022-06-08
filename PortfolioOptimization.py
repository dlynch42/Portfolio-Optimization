from datetime import date
import pandas_datareader.data as web
import quandl
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


quandl.ApiConfig.api_key = 'wmGcAuhm3L84uykaZWyj'  # need the api key at the beginning to be able to call enough

start = pd.to_datetime('2018-1-01')
end = pd.to_datetime(date.today())

spy = web.DataReader('SPY', 'yahoo', start, end)
amzn = web.DataReader('AMZN', 'yahoo', start, end)
aapl = web.DataReader('AAPL', 'yahoo', start, end)
tsla = web.DataReader('TSLA', 'yahoo', start, end)
goog = web.DataReader('GOOG', 'yahoo', start, end)
jpm = web.DataReader('JPM', 'yahoo', start, end)
halo = web.DataReader('HALO', 'yahoo', start, end)
uber = web.DataReader('UBER', 'yahoo', start, end)
abnb = web.DataReader('ABNB', 'yahoo', start, end)
sofi = web.DataReader('SOFI', 'yahoo', start, end)
pcg = web.DataReader('PCG', 'yahoo', start, end)
t = web.DataReader('T', 'yahoo', start, end)

for stock_df in (spy, amzn, aapl, tsla, goog, jpm, halo, uber, abnb, sofi, pcg, t):
    stock_df['Normed Return'] = stock_df['Close'] / stock_df.iloc[0]['Close']

for stock_df, allo in zip((spy, amzn, aapl, tsla, goog, jpm, halo, uber, abnb, sofi, pcg, t),
                          [(1/12), (1/12), (1/12), (1/12), (1/12), (1/12), (1/12), (1/12), (1/12), (1/12), (1/12), (1/12), (1/12)]):
    stock_df['Allocation'] = stock_df['Normed Return'] * allo

for stock_df in (spy, amzn, aapl, tsla, goog, jpm, halo, uber, abnb, sofi, pcg, t):
    stock_df['Position Values'] = stock_df['Allocation'] * 4106.93

all_pos_values = [spy['Position Values'], amzn['Position Values'], aapl['Position Values'], tsla['Position Values'],
                  goog['Position Values'], jpm['Position Values'], halo['Position Values'], uber['Position Values'],
                  abnb['Position Values'], sofi['Position Values'], pcg['Position Values'], t['Position Values']]
portfolio_val = pd.concat(all_pos_values, axis=1)

portfolio_val.columns = ['SPY', 'AMZN', 'AAPL', 'TSLA', 'GOOG', 'JPM', 'HALO', 'UBER', 'ABNB', 'SOFI', 'PCG', 'T']
portfolio_val['Total Position'] = portfolio_val.sum(axis=1)

portfolio_val['Total Position'].plot(figsize=(10, 8))
plt.title('Total Stock Portfolio Value')
plt.show()

portfolio_val.drop('Total Position', axis=1).plot(figsize=(10, 8))
plt.show()

portfolio_val['Daily Return'] = portfolio_val['Total Position'].pct_change(1)
portfolio_val['Daily Return'].plot(kind='hist', bins=100, figsize=(4, 5))
plt.show()

portfolio_val['Daily Return'].plot(kind='kde', figsize=(4, 5))
plt.show()

cum_return = 100 * (portfolio_val['Total Position'][-1]/portfolio_val['Total Position'][0] - 1)

SR = portfolio_val['Daily Return'].mean()/portfolio_val['Daily Return'].std()

ASR = (252 ** 0.5) * SR

stocks = pd.concat([spy['Adj Close'], amzn['Adj Close'], aapl['Adj Close'], tsla['Adj Close'],
                    goog['Adj Close'], jpm['Adj Close'], halo['Adj Close'], uber['Adj Close'],
                    abnb['Adj Close'], sofi['Adj Close'], pcg['Adj Close'], t['Adj Close']], axis=1)
stocks.columns = ['SPY', 'AMZN', 'AAPL', 'TSLA', 'GOOG', 'JPM', 'HALO', 'UBER', 'ABNB', 'SOFI', 'PCG', 'T']

# log returns normalizes returns
log_ret = np.log(stocks/stocks.shift(1))

# Monte Carlo
np.random.seed(101)

num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    # weights
    weights = np.array(np.random.random(12))  # needs to be the same number of stocks
    weights = weights/np.sum(weights)

    # Save weights
    all_weights[ind, :] = weights

    # Expected Return
    ret_arr[ind]=np.sum((log_ret.mean() * weights) * 252)
    # Expected Volatility
    vol_arr[ind]=np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    # Sharpe Ratio
    sharpe_arr[ind]=ret_arr[ind]/vol_arr[ind]

# Best SR
sharpe_arr.argmax()
# print(all_weights[1420,:]); # shows array
max_sr_ret=ret_arr[1420]
max_sr_vol=vol_arr[1420]

plt.figure(figsize=(12, 8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.title('Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')  # Shows optimal SR
plt.show()


# Mathematical Optimization
def get_ret_vol_sr(weight):
    weight = np.array(weight)
    ret = np.sum(log_ret.mean() * weight) * 252
    vol = np.sqrt(np.dot(weight.T, np.dot(log_ret.cov() * 252, weight)))
    sr = ret/vol
    return np.array([ret, vol, sr])


# Minimize SR
def neg_sharpe(weight):
    return get_ret_vol_sr(weight)[2] * -1


def check_sum(weight):
    # return 0 if sum of weights is 1, or how far off
    return np.sum(weight) - 1


# create constraint
cons = ({'type': 'eq', 'fun': check_sum})
bounds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),)
init_guess = [0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333,
            0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333]
# init_guess needs to be a decimal and has to add up to 1
opt_results = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds,
                       constraints=cons)  # Sequential least squared
print(opt_results.x)    # optimal results

frontier_y=np.linspace(0, 0.3, 100)


def minimize_volatility(weight):
    return get_ret_vol_sr(weight)[1]


frontier_volatilty = []

for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun': check_sum},
          {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})  # some weight (w), get first return - possible return

    result = minimize(minimize_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

    frontier_volatilty.append(result['fun'])

# Marcowitz Portfolio Optimization
plt.figure(figsize=(12, 8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.plot(frontier_volatilty, frontier_y, 'g--', linewidth=3)
plt.show()  # shows for each level of volatility, what is the best possible return you can get