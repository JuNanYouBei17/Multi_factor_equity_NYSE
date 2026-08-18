import pandas as pd
import numpy as np
import pathlib
import csv

#%%
"""
Load the three datasets. For prices-split-adjusted.csv:

Parse dates, sort ascending
Report how many unique tickers and trading days are in the dataset
Identify any tickers with more than 20% missing closing prices across the full date range, and drop them
"""
path = pathlib.Path("/Users/zengnan/Desktop/2027 Summer Internship/Coding Interview/Data/New York Stock Exchange/")
prices = pd.read_csv(path / "prices.csv")
fundamentals = pd.read_csv(path / "fundamentals.csv")
securities = pd.read_csv(path / "securities.csv")
prices_split_adj = pd.read_csv(path / "prices-split-adjusted.csv",parse_dates=['date'])

prices_split_adj.sort_values(by='date',ascending=True,inplace=True)
tickers_unique = prices_split_adj['symbol'].nunique()
trading_days_unique = prices_split_adj['date'].nunique()
print(f'unique tickers:{tickers_unique}')
print(f'unique trading days:{trading_days_unique}')

# ~mask selects columns where missing ratio <= 20% (i.e. mask is False)
closing_price_df = pd.pivot_table(prices_split_adj,index='date',columns='symbol',values='close')
mask = closing_price_df.isna().sum(axis=0) > closing_price_df.shape[0] * 0.2
filtered_close_price = closing_price_df.loc[:, ~mask]


#%%
'''
Construct a daily close price matrix (dates × tickers). 
Forward-fill missing values, but only up to a maximum of 5 consecutive days.
Explain in a comment why uncapped forward-fill is dangerous in a real production system.
'''
daily_close_price = filtered_close_price.ffill(limit=5)
# ffill will propagate the last known price indefinitely,
# making a dead stock appear alive in the portfolio.

# %% Section 2 — Factor Construction (25 min)
"""
Compute the following return series from the price matrix:
- 1-day return
- 5-day return
- 21-day return
"""
returns_daily = daily_close_price.pct_change(periods=1)
weekly_daily = daily_close_price.pct_change(periods=5)
monthly_daily = daily_close_price.pct_change(periods=21)


# %%
"""
Construct the following signals from fundamentals.csv:

Apply a 90-day publication lag (publish_date = Period Ending + 90 days)
Calculate the four signals below

| Signal | Formula |
|---|---|
| ROE | `Net Income / Total Equity` |
| Asset Turnover | `Total Revenue / Total Assets` |
| Earnings Yield | `Earnings Per Share / Close Price` |
| Cash Flow Quality | `Net Cash Flow-Operating / Net Income` |

For Earnings Yield, merge with your price matrix to get the close price on publish_date 
— describe the join logic in comments
Pivot each signal into a matrix (dates × tickers) using publish_date as index
Reindex to your daily price matrix index and ffill 
— this carries each annual filing forward until the next one arrives

"""
fundamentals['Period Ending'] = pd.to_datetime(fundamentals['Period Ending'])
fundamentals['Published Date'] = fundamentals['Period Ending'] + pd.Timedelta(90, unit='D')
# Need to add unit for timedelta to avoid ambiguity

fundamentals['ROE'] = fundamentals['Net Income'] / fundamentals['Total Equity']
fundamentals['Asset Turnover'] = fundamentals['Total Revenue'] / fundamentals['Total Assets']
fundamentals['Cash Flow Quality'] = fundamentals['Net Cash Flow-Operating'] / fundamentals['Net Income']

# Left join fundamentals onto melted prices on (publish_date == date) and (Ticker Symbol == symbol)
melted_close_price = filtered_close_price.reset_index().melt(id_vars='date', var_name='symbol', value_name='close')
fundamentals_new = pd.merge(fundamentals,melted_close_price, left_on=['Published Date', 'Ticker Symbol'], right_on=['date', 'symbol'], how='left')

# Earnings Yield = EPS / close price on publish_date,
# capturing the valuation at the time the market could first see the filing.
fundamentals_new['Earnings Yield'] = fundamentals_new['Earnings Per Share'] / fundamentals_new['close']


def pivot_reindex_signal(fundamentals_new, signal_name):

    signal_df = pd.pivot_table(fundamentals_new,index='Published Date',columns='Ticker Symbol',values=signal_name)
    signal_df_reindexed = signal_df.reindex(daily_close_price.index).ffill()
    return signal_df_reindexed

roe = pivot_reindex_signal(fundamentals_new, 'ROE')
asset_turnover = pivot_reindex_signal(fundamentals_new, 'Asset Turnover')
earnings_yield = pivot_reindex_signal(fundamentals_new, 'Earnings Yield')
cash_flow_quality = pivot_reindex_signal(fundamentals_new, 'Cash Flow Quality')


# %%
"""
Construct a 12M-1M momentum signal and a 1M reversal signal. 
Explain the economic intuition behind each in a comment block.

"""
mom_12_1 = daily_close_price.shift(21) / daily_close_price.shift(252) - 1
# 12M-1M momentum captures the cumulative return from 12 months ago to 1 month ago.

reversal_1m = daily_close_price / daily_close_price.shift(21) - 1
# 1M reversal captures the return from 1 month ago to today.



# %%
"""
Write a function `winsorise(df, pct=0.01)` 
caps values at the 1st and 99th percentile cross-sectionally (per date). 
Apply it to all six signals.
"""
def winsorise(df, pct=0.01):
    df_winsorise = df.apply(lambda x: x.clip(lower=x.quantile(pct),
                              upper=x.quantile(1-pct)),axis=1)
    return df_winsorise

roe_winsorised = winsorise(roe)
asset_turnover_winsorised = winsorise(asset_turnover)
earnings_yield_winsorised = winsorise(earnings_yield)
cash_flow_quality_winsorised = winsorise(cash_flow_quality)
mom_12_1_winsorised = winsorise(mom_12_1)
reversal_1m_winsorised = winsorise(reversal_1m)
    
# %%
"""
Write a function `sector_neutralise(signal_df, sector_map)` 
demeanes each signal by GICS sector per date. 
Verify your neutralisation worked by printing the mean sector exposure before and after.
"""

sector_map = securities[['Ticker symbol','GICS Sector']].set_index('Ticker symbol') 

def sector_neutralise(signal_df, sector_map):
    signal_copy = signal_df.copy()  # 保护原始数据
    asset_col = signal_copy.columns.to_list()
    signal_copy.columns = signal_copy.columns.map(sector_map['GICS Sector'])
    # groupby along axis=1 (columns direction) — group tickers by their sector
    # We use .transform('mean') because it broadcasts the sector average 
    # back to the original dimensions of the dataframe. 
    # This allows for a direct element-wise subtraction (Stock Signal - Sector Mean) 
    # while maintaining the 388-column structure.
    sector_average = signal_copy.groupby(signal_copy.columns, axis=1).transform('mean')
    demeaned = signal_copy - sector_average
    demeaned.columns = asset_col
    return demeaned

roe_neutralised = sector_neutralise(roe_winsorised, sector_map)
asset_turnover_neutralised = sector_neutralise(asset_turnover_winsorised, sector_map)
earnings_yield_neutralised = sector_neutralise(earnings_yield_winsorised, sector_map)
cash_flow_quality_neutralised = sector_neutralise(cash_flow_quality_winsorised, sector_map)
mom_12_1_neutralised = sector_neutralise(mom_12_1_winsorised, sector_map)
reversal_1m_neutralised = sector_neutralise(reversal_1m_winsorised, sector_map)

def sector_exposure(signal_df, sector_map):
    signal_copy = signal_df.copy()
    signal_copy.columns = signal_copy.columns.map(sector_map['GICS Sector'])
    exposure = signal_copy.groupby(signal_copy.columns, axis=1).mean()
    return exposure

sector_exposure_before = sector_exposure(roe_winsorised, sector_map)
sector_exposure_after = sector_exposure(roe_neutralised, sector_map)

print("Sector exposure before neutralisation:",sector_exposure_before.mean())
print("Sector exposure after neutralisation:",sector_exposure_after.mean())

# %%
"""
Convert all six processed signals into cross-sectional z-scores. 

Select common stocks to ensure signals are based on the same universe.
Then construct a composite score with the following weights:

| Signal | Weight |
|---|---|
| Momentum 12M-1M | 35% |
| ROE | 15% |
| Asset Turnover | 15% |
| Earnings Yield | 15% |
| Cash Flow Quality | 10% |
| Reversal 1M | 10% |
"""
def cross_sectional_zscore(df):
    return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1),axis=0)

mom_12_1_zscore = cross_sectional_zscore(mom_12_1_neutralised)
roe_zscore = cross_sectional_zscore(roe_neutralised)
asset_turnover_zscore = cross_sectional_zscore(asset_turnover_neutralised)
earnings_yield_zscore = cross_sectional_zscore(earnings_yield_neutralised)
cash_flow_quality_zscore = cross_sectional_zscore(cash_flow_quality_neutralised)
reversal_1m_zscore = cross_sectional_zscore(reversal_1m_neutralised)

common_cols = (mom_12_1_zscore.columns
               .intersection(roe_zscore.columns)
               .intersection(asset_turnover_zscore.columns)
               .intersection(earnings_yield_zscore.columns)
               .intersection(cash_flow_quality_zscore.columns)
               .intersection(reversal_1m_zscore.columns))

"""
composite_score = (mom_12_1_zscore[common_cols] * 0.35 +
                    roe_zscore[common_cols] * 0.15 + 
                    asset_turnover_zscore[common_cols] * 0.15 +
                    earnings_yield_zscore[common_cols] * 0.15 +
                    cash_flow_quality_zscore[common_cols] * 0.10 +
                    reversal_1m_zscore[common_cols] * 0.10)
"""
composite_score = (
    mom_12_1_zscore[common_cols] * 0.60 +
    asset_turnover_zscore[common_cols] * 0.20 +
    earnings_yield_zscore[common_cols] * 0.10 +
    reversal_1m_zscore[common_cols] * 0.10
)

# %% ## Section 4 — Portfolio Construction (15 min)

"""
On the first trading day of each month, rank all stocks by composite score. 
Construct an equal-weighted long-short portfolio:
- Long: top quintile
- Short: bottom quintile
- Verify the portfolio is dollar neutral
(long weight sum = short weight sum in absolute terms)
"""
month_start = composite_score.resample('MS').first().index
idx = composite_score.index.intersection(month_start)
composite_score_monthly = composite_score.loc[idx]

def rank_signal_monthly(composite_score):

    month_start = composite_score.resample('MS').first().index
    idx = composite_score.index.intersection(month_start)
    composite_score_monthly = composite_score.loc[idx]

    ranked_monthly_score = composite_score_monthly.rank(axis=1, ascending=True, pct=True)
    return ranked_monthly_score

ranked_monthly_score = rank_signal_monthly(composite_score)

# Drop rows where all values are NaN, which can occur if no stocks have valid scores at the start of the month.
ranked_monthly_score.dropna(how='all', inplace=True)

def construct_trading_signal(df, pct=0.2):
    position_signal = pd.DataFrame(index=ranked_monthly_score.index, columns=ranked_monthly_score.columns)
    position_signal[ranked_monthly_score >= 1-pct] = 1  # Long
    position_signal[ranked_monthly_score <= pct] = -1 # Short
    position_signal.fillna(0, inplace=True) # Neutral
    return position_signal

position_signal = construct_trading_signal(ranked_monthly_score, pct=0.2)

def construct_ew_portfolio(position_signal):
   nlong = (position_signal==1).sum(axis=1)
   nshort = (position_signal==-1).sum(axis=1)
   long_weights = position_signal.where(position_signal==1).div(nlong,axis=0)
   short_weights = position_signal.where(position_signal==-1).div(nshort,axis=0)
   # Fillna before summing, as NA cannot perform calculation
   weights_df = long_weights.fillna(0) + short_weights.fillna(0)

   return weights_df

weights = construct_ew_portfolio(position_signal)

weights.where(weights > 0).sum(axis=1) # Long weight sum should be 1
weights.where(weights < 0).sum(axis=1).abs() # Short weight sum should be 1


# %% Section 5 — Backtesting & Evaluation (15 min)
"""
Calculate daily net portfolio returns with **10bps one-way transaction cost**. 
Compute:
- Annualised Sharpe Ratio
- Maximum Drawdown
- Average monthly turnover
- Annualised return and volatility separately
"""

# Reindex weights to daily frequency using ffill
weights_daily = weights.reindex(returns_daily.index).ffill()
weights_daily.dropna(how='all', inplace=True) 
col = weights.columns.intersection(returns_daily.columns)
idx = weights_daily.index.intersection(returns_daily.index)
gross_returns = (weights_daily.loc[idx,col].shift(1) * returns_daily.loc[idx,col]).sum(axis=1)

# Turnover = absolute change in weights, divide by 2 to avoid double counting
turnover = weights_daily.diff().abs().sum(axis=1) / 2 
turnover_cost = turnover * 0.001

net_returns = gross_returns - turnover_cost

def sharpe(returns):
    return returns.mean() / returns.std() * np.sqrt(252)

def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()  # 需要加1来计算累计收益率
    peak = cumulative.cummax()
    drawdown = (cumulative - peak)/peak
    return drawdown.min()

def annualised_return(returns):
    return (1 + returns.mean()) ** 252 - 1 

def annualised_vol(returns):
    return returns.std() * np.sqrt(252)

def monthly_turnover(weights_daily):
    daily_turnover = (weights_daily.diff().abs()/2).sum(axis=1)
    monthly_turnover = daily_turnover.resample('M').sum()
    return monthly_turnover

def compute_statistics(returns, weights_daily):
    stats = {}
    stats['Annualised Return'] = annualised_return(returns)
    stats['Annualised Volatility'] = annualised_vol(returns)
    stats['Sharpe Ratio'] = sharpe(returns)
    stats['Max Drawdown'] = max_drawdown(returns)
    stats['Monthly Turnover'] = monthly_turnover(weights_daily).mean()
    return pd.DataFrame(stats,index=['stats']).T


summary_stats = compute_statistics(net_returns, weights_daily)
print(summary_stats)

#%%

"""
Compute the monthly IC (Spearman rank correlation) 
between your composite score and the subsequent 21-day forward return. 
Report the ICIR (mean IC / std IC) and its t-statistic. 
Interpret the result in one sentence.

# Information Coefficient (IC) measures the predictive power of a signal
# by calculating the cross-sectional Spearman rank correlation between
# current signal values and future asset returns.
#
# A positive IC indicates that assets with higher signal values tend to
# outperform in the future, while a negative IC suggests the opposite
"""
from scipy.stats import spearmanr

forward_returns_21d = daily_close_price.shift(-21) / daily_close_price - 1

composite_score_monthly.dropna(how='all', inplace=True)
common_index = composite_score_monthly.index.intersection(forward_returns_21d.index)
common_cols = composite_score_monthly.columns.intersection(forward_returns_21d.columns)

forward_returns_monthly = forward_returns_21d.loc[common_index, common_cols]
composite_score_monthly_aligned = composite_score_monthly.loc[common_index, common_cols]

def compute_ic(composite_score_monthly_aligned, forward_returns_monthly):
    ic_values = {}
    for i in composite_score_monthly_aligned.index:
        score = composite_score_monthly_aligned.loc[i].dropna()
        fwd_ret = forward_returns_monthly.loc[i].dropna()

         # Select common stocks（not column because series after dropna() has only one dimension)
        common_tickers = score.index.intersection(fwd_ret.index)
        if len(common_tickers) < 10:
            continue

        ic, pvalue = spearmanr(score[common_tickers], fwd_ret[common_tickers])
        ic_values[i] = ic

    return pd.DataFrame(ic_values, index=['IC']).T

ic_df = compute_ic(composite_score_monthly_aligned, forward_returns_monthly)
icir = ic_df['IC'].mean()/ ic_df.std()


n = len(ic_df)
t_stat = icir * np.sqrt(n)


print(f"ICIR: {icir}")
print(f"T-statistic: {t_stat}")

# The strategy yields a mean IC of -0.0235 with a t-statistic of -0.72,
#  indicating that the composite score has no statistically significant predictive power over the subsequent 21-day returns. 
# The low ICIR suggests that the signal is dominated by noise, 
# and we cannot reject the null hypothesis that the true predictive rank correlation is zero

# %%## Section 6 — Visualisation & Diagnosis (5 min)
'''
Produce the following three plots in a single figure:
1. Cumulative portfolio index (starting at 100)
2. Rolling 12-month Sharpe Ratio
3. Monthly IC over time with a horizontal line at IC = 0

Q14.Plot a bar chart of average sector exposure of the final portfolio. 
What does this tell you about whether your sector neutralisation in Q10 was effective?
'''

import matplotlib.pyplot as plt

portfolio_index = 100*(1+net_returns).cumprod()
rolling_sharpe = net_returns.rolling(window=252).apply(sharpe)

plt.figure(figsize=(15,10))
plt.subplot(3,1,1)
portfolio_index.plot(title='Cumulative Portfolio Index')
plt.subplot(3,1,2)
rolling_sharpe.plot(title='Rolling 12-Month Sharpe Ratio')
plt.subplot(3,1,3)
ic_df['IC'].plot(title='Monthly IC Over Time')
plt.axhline(0, color='red', linestyle='--')
plt.tight_layout()

final_exposure = weights.abs()/2
final_exposure.columns = final_exposure.columns.map(sector_map['GICS Sector'])

# We use .sum() to aggregate the individual stock weights into a single value 
# per sector. This reduces the dimensionality (e.g., from 388 stocks to 11 sectors)
avg_sector_exposure = final_exposure.groupby(final_exposure.columns,axis=1).sum().mean()
avg_sector_exposure.plot(kind='bar', title='Average Sector Exposure')
plt.show()


# %% Bonues
"""
Split the backtest into two sub-periods (pre/post 2014). 
Compare Sharpe and ICIR across periods. What might explain any difference?
"""

net_returns_pre_2014 = net_returns[net_returns.index < '2014-01-01']
net_returns_post_2014 = net_returns[net_returns.index >= '2014-01-01']

sharpe_pre_2014 = sharpe(net_returns_pre_2014)
sharpe_post_2014 = sharpe(net_returns_post_2014)

ic_pre_2014 = ic_df[ic_df.index < '2014-01-01']['IC']
ic_post_2014 = ic_df[ic_df.index >= '2014-01-01']['IC']

print(f"Sharpe Ratio Pre-2014: {sharpe_pre_2014}")
print(f"Sharpe Ratio Post-2014: {sharpe_post_2014}")
print(f"ICIR Pre-2014: {ic_pre_2014.mean()/ic_pre_2014.std()}")
print(f"ICIR Post-2014: {ic_post_2014.mean()/ic_post_2014.std()}")

# %% 

"""
Backtest each of the six signals individually. 
Report standalone Sharpe and IC for each. 
Does the signal weighting in Q8 look justified?
"""

def backtest_single_signal(signal_zscore, signal_name, returns_daily):
    # Portfolio construction
    rank = signal_zscore.resample('MS').first().rank(axis=1, ascending=True, pct=True)
    rank.dropna(how='all', inplace=True)
    
    position = pd.DataFrame(0, index=rank.index, columns=rank.columns)
    position[rank >= 0.8] = 1
    position[rank <= 0.2] = -1
    
    nlong = (position == 1).sum(axis=1)
    nshort = (position == -1).sum(axis=1)
    long_w = position.where(position == 1).div(nlong, axis=0).fillna(0)
    short_w = position.where(position == -1).div(nshort, axis=0).fillna(0)
    weights = long_w + short_w
    
    # Daily returns
    col = weights.columns.intersection(returns_daily.columns)
    weights_daily = weights.reindex(returns_daily.index).ffill()
    gross_ret = (weights_daily[col].shift(1) * returns_daily[col]).sum(axis=1)
    tc = weights_daily[col].diff().abs().sum(axis=1) / 2 * 0.001
    net_ret = gross_ret - tc
    
    # IC
    fwd_ret = daily_close_price.shift(-21) / daily_close_price - 1
    fwd_monthly = fwd_ret.reindex(rank.index)
    ic_vals = {}
    for date in rank.index:
        s = signal_zscore.resample('MS').first().loc[date].dropna()
        f = fwd_monthly.loc[date].dropna() if date in fwd_monthly.index else pd.Series()
        common = s.index.intersection(f.index)
        if len(common) >= 10:
            ic, _ = spearmanr(s[common], f[common])
            ic_vals[date] = ic
    ic_series = pd.Series(ic_vals)
    
    return {
        'Signal': signal_name,
        'Sharpe': sharpe(net_ret),
        'Mean IC': ic_series.mean(),
        'ICIR': ic_series.mean() / ic_series.std()
    }

# Run all six
signals = {
    'Momentum': mom_12_1_zscore,
    'ROE': roe_zscore,
    'Asset Turnover': asset_turnover_zscore,
    'Earnings Yield': earnings_yield_zscore,
    'Cash Flow Quality': cash_flow_quality_zscore,
    'Reversal': reversal_1m_zscore
}

results = [backtest_single_signal(z, name, returns_daily) for name, z in signals.items()]
pd.DataFrame(results).set_index('Signal')



# %%
