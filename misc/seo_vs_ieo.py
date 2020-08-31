import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

df = pd.read_pickle('results/numerical_study.pkl')
df = df[['p_spot', 'p_m1', 'ridge', 'lasso', 'mlp_seo', 'rnn_seo', 'lstm_seo', 'dda_ml1', 'dda_ml2', 'mlp_ieo', 'rnn_ieo', 'lstm_ieo']]
df.columns = ['P-SPOT', 'P-M1', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM']

df.columns = pd.MultiIndex.from_arrays([['Baseline'] * 2 + ['SEO'] * 5 + ['IEO'] * 5, df.columns])

df = df.iloc[:, 2:]

plt.figure(figsize=(5, 5))
ax = plt.gca()

df = df.loc[pd.IndexSlice['rw', True, 10, :, 73], :]

for alg in df.columns.get_level_values(1).unique():
    if alg == 'LSTM':
        plt.scatter(df.loc[:, ('SEO', alg)].to_numpy(), df.loc[:, ('IEO', alg)].to_numpy(), s=10, label=alg)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', lw=1, zorder=1)
plt.axis('square')
plt.legend()
plt.show()
# plt.savefig('misc/scatter.pgf')