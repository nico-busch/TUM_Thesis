import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

tum1 = (100/255, 160/255, 200/255)
tum2 = (0, 101 / 255, 189 / 255)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Prepare dataframes
df = pd.read_csv('data/gas.csv', parse_dates=['Date'], index_col='Date')
df['Day'] = df.index.to_period('D').where(df.index.hour >= 6,
                                          (df.index - pd.DateOffset(days=1)).to_period('D'))
df['Month'] = df['Day'].dt.asfreq('M')
df['Daily'] = df.groupby('Day')['Demand'].transform('mean')
df['Level'] = df.groupby('Month')['Daily'].transform('min')
daily = df.groupby('Day')['Demand'].mean()
daily = daily.to_timestamp()
level = df.groupby('Month')['Daily'].min()
level = level.to_timestamp()
val = level.iloc[[-1]]
val.index = val.index + pd.DateOffset(months=1) - pd.DateOffset(days=1)
level = level.append(val)

# Plot demand
plt.figure(figsize=(14 / 2.54, 3))
ax = plt.gca()
peak = plt.fill_between(daily.index, daily, color=tum1, step='post', label='Peak Load')
base = plt.fill_between(level.index, level, color=tum2, step='post', label='Base Load')
ax.autoscale(axis='x', tight=True)
plt.ylim(bottom=0)
plt.ylabel('MW')
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.2)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('pgf/demand.pgf')

