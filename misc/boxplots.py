import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


df = pd.read_pickle('results/linear.pkl')
df = df[['p_spot', 'p_m1', 'ridge', 'lasso', 'mlp_seo', 'rnn_seo', 'lstm_seo', 'dda_ml1', 'dda_ml2', 'mlp_ieo', 'rnn_ieo', 'lstm_ieo']]
df.columns = ['P-Spot', 'P-M1', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM']

df.columns = pd.MultiIndex.from_arrays([['Baseline'] * 2 + ['SEO'] * 5 + ['IEO'] * 5, df.columns])

test = df.loc[pd.IndexSlice[:, :, :, 48], :]

fig, axes = plt.subplots(figsize=(6, 6 / 16 * 9), ncols=3, nrows=1, sharey=True, gridspec_kw={'width_ratios': [2, 5, 5]})

for label, ax in zip(df.columns.get_level_values(0).unique(), axes):

    bp = test.loc[:, label].boxplot(ax=ax, rot=90, return_type='dict', patch_artist=True, widths=0.5)

    for element in ['boxes', 'whiskers', 'means', 'caps']:
        plt.setp(bp[element], color='black')

    for element in ['fliers']:
        plt.setp(bp[element], markeredgecolor=(0, 101/255, 189/255))

    for element in ['medians']:
        plt.setp(bp[element], color=(0, 101/255, 189/255), linewidth=2)

    for patch in bp['boxes']:
        patch.set(facecolor='white')

    ax.grid(False, which='both', axis='x')
    ax.set_xlabel(label)

plt.ylim(bottom=0)
plt.subplots_adjust(bottom=0.2)
fig.subplots_adjust(wspace=0)




# plt.show()

plt.savefig('results/boxplots.pgf')
