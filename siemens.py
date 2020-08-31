import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('gas.csv', parse_dates=['Date'])

df['Month'] = df['Date'].dt.to_period('M').where((df['Date'].dt.day > 1) | (df['Date'].dt.hour >= 6),
                                                 (df['Date'] - pd.DateOffset(months=1)).dt.to_period('M'))
spot = df.groupby('Month')['Spot'].mean().to_numpy()

cost = df.groupby('Month')['Cost'].sum().to_numpy()

print((df['Cost']).sum())
exit()

df2 = pd.read_csv('data/reuters.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)
df2.index.freq = 'M'

spot2 = df2.loc['2017-01-01':, 'SPOT'].to_numpy()

forward = df2.loc['2017-01-01':, 'M1'].to_numpy()

print(cost.mean())

exit()

