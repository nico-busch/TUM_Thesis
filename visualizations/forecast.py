import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

tum1 = (0, 101 / 255, 189 / 255)
tum2 = (227/255, 114/255, 34/255)
tum3 = (162/255, 173/255, 0)
tum4 = (51/255, 51/255, 51/255)
tum5 = (100/255, 160/255, 200/255)
tum6 = (0/255, 51/255, 89/255)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Prepare dataframes
fc = pd.read_pickle('results/forecast.pkl')
df = pd.read_csv('data/reuters.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)
results = pd.DataFrame(columns=fc.columns[1:])

# Calculate predictive performance metrics
results.loc['RMSE'] = (fc.iloc[:, 1:].subtract(fc['ACTUAL'], axis=0) ** 2).mean() ** 0.5
results.loc['SMAPE'] = (2 * fc.iloc[:, 1:].subtract(fc['ACTUAL'], axis=0).abs()).divide(
    fc.iloc[:, 1:].abs().add(fc['ACTUAL'].abs(), axis=0)).mean()
results.loc['MASE'] = (fc.iloc[:, 1:].subtract(fc['ACTUAL'], axis=0).abs()).mean().divide(
    (fc['NAIVE'].subtract(fc['ACTUAL']).abs()).mean(), axis=0)
print(results)

# Plot forecast curves
plt.figure(figsize=(14 / 2.54, 3))
ax = plt.gca()
ax.plot(fc['ACTUAL'], label='SPOT', color=tum1, lw=3)
ax.plot(fc['LASSO'], label='LASSO', color=tum5, lw=1)
ax.plot(fc['Ridge'], label='RIDGE', color=tum6, lw=1)
ax.plot(fc['MLP'], label='MLP-SEO', color=tum4,  lw=1)
ax.plot(fc['RNN'], label='RNN-SEO', color=tum2, lw=1)
ax.plot(fc['LSTM'], label='LSTM-SEO', color=tum3, lw=1)
ax.yaxis.grid()
ax.autoscale(axis='x', tight=True)
plt.xticks(rotation=45, ha="right")
plt.ylabel('€/MWh')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('pgf/forecast.pgf')