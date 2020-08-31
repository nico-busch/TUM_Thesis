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

tum1 = (0, 101 / 255, 189 / 255)
tum2 = (227/255, 114/255, 34/255)

df = pd.read_pickle('results/numerical_study.pkl')
df = df[['p_spot', 'p_m1', 'ridge', 'lasso', 'mlp_seo', 'rnn_seo', 'lstm_seo', 'dda_ml1', 'dda_ml2', 'mlp_ieo', 'rnn_ieo', 'lstm_ieo']]
df.columns = ['P-SPOT', 'P-M1', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM']

df.columns = pd.MultiIndex.from_arrays([['Baseline'] * 2 + ['SEO'] * 5 + ['IEO'] * 5, df.columns])


test = df.loc[pd.IndexSlice[:, :, 10, :, 73], :]

fig, axes = plt.subplots(figsize=(14/2.54, 9), ncols=3, nrows=4, sharey='row', gridspec_kw={'width_ratios': [2, 5, 5]})

for i, (row, params) in enumerate(zip(axes, pd.MultiIndex.from_product([df.index.get_level_values(0).unique(),
                                                                  df.index.get_level_values(1).unique()]))):

    for label, ax in zip(df.columns.get_level_values(0).unique(), row):

        bp = test.loc[params, label].boxplot(ax=ax, rot=90, return_type='dict', patch_artist=True, widths=0.5,
                                            showmeans=True, meanline=True)

        for element in ['boxes', 'whiskers', 'means', 'caps']:
            plt.setp(bp[element], color=tum1)

        for element in ['fliers']:
            plt.setp(bp[element], markerfacecolor=tum1, markeredgecolor=tum1, markersize=3)

        for element in ['medians']:
            plt.setp(bp[element], color=(0, 101/255, 189/255), linewidth=2)

        for element in ['means']:
            plt.setp(bp[element], color=tum2, linewidth=2, ls='-')

        for patch in bp['boxes']:
            patch.set(alpha=None, facecolor=tum1 + (0.5,), edgecolor=tum1)

        ax.grid(False, which='both', axis='x')
        if i != 3:
            ax.tick_params(labelbottom=False)
        if i == 3:
            ax.set_xlabel(label)

        ax.set_ylim(bottom=0)
        ax.yaxis.get_major_locator().set_params(integer=True)

    row[0].set_ylabel('Test PE in %')
    string = ('MR' if params[0] == 'mr' else 'RW') + ', ' + ('Linear' if not params[1] else 'Logarithmic')
    row[2].text(0.95, 0.95, string, horizontalalignment='right', verticalalignment='top',
                transform=row[2].transAxes, zorder=3)

plt.tight_layout()
fig.subplots_adjust(wspace=0)

# plt.show()

plt.savefig('results/boxplots.pgf')
