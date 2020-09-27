import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

tum1 = (0, 101 / 255, 189 / 255)
tum2 = (227/255, 114/255, 34/255)
tum3 = (162/255, 173/255, 0)
tum4 = (51/255, 51/255, 51/255)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Prepare dataframes
df = pd.read_pickle('results/runtimes.pkl')
df = df.astype(float)
df = df.groupby('train_size').mean()

# Plot runtimes
plt.figure(figsize=(14 / 2.54, 3))
ax = plt.gca()
ax.plot(df['dda_bd'], label='DDA-BD', color=tum4, marker='o', markersize=3)
ax.plot(df['mlp_ieo'], label='MLP-IEO', color=tum1, marker='o', markersize=3)
ax.plot(df['rnn_ieo'], label='RNN-IEO', color=tum2, marker='o', markersize=3)
ax.plot(df['lstm_ieo'], label='LSTM-IEO', color=tum3, marker='o', markersize=3)
ax.autoscale(axis='x', tight=True)
loc = plticker.MultipleLocator(base=20)
ax.xaxis.set_major_locator(loc)
ax.yaxis.grid()
ax.set_yscale('log')
plt.ylabel('Runtime [s]')
plt.xlabel('Training set size $T$')
plt.legend()
plt.tight_layout()
plt.savefig('pgf/runtimes.pgf')
plt.show()