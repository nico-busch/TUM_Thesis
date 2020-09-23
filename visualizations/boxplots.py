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

# Prepare dataframes
df = pd.read_pickle('results/numerical_study.pkl')
df = df[['p_spot', 'p_m1', 'lasso', 'ridge', 'mlp_seo', 'rnn_seo', 'lstm_seo', 'dda_ml1', 'dda_ml2', 'mlp_ieo', 'rnn_ieo', 'lstm_ieo']]
df.columns = ['P-SPOT', 'P-M1',
              'LASSO', 'RIDGE', 'MLP-SEO', 'RNN-SEO', 'LSTM-SEO',
              'DDA-ML1', 'DDA-ML2', 'MLP-IEO', 'RNN-IEO', 'LSTM-IEO']
df.columns = pd.MultiIndex.from_arrays([['Baseline'] * 2 + ['Sequential'] * 5 + ['Integrated'] * 5, df.columns])
test = df.loc[pd.IndexSlice[:, :, 10, :, 73], :]

# Plot boxplots
fig, axes = plt.subplots(figsize=(14/2.54, 9.25), ncols=3, nrows=4, sharey='row', gridspec_kw={'width_ratios': [2, 5, 5]})
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
            ax.set_xlabel(r'\textsc{' + label + '}')
            ax.xaxis.set_label_coords(0.5, -0.5)

        ax.set_ylim(bottom=0)
        ax.yaxis.get_major_locator().set_params(integer=True)

    string = ('MR' if params[0] == 'mr' else 'RW') + ' (' + ('linear' if not params[1] else 'logarithmic') + ')'
    row[0].set_ylabel('Test PE in %')
    row[2].set_ylabel(string, labelpad=10)
    row[2].yaxis.set_label_position('right')

axes[2][0].set_ylim(*axes[3][0].get_ylim())
axes[1][0].set_ylim(*axes[0][0].get_ylim())

plt.tight_layout()
fig.subplots_adjust(wspace=0)
plt.show()
plt.savefig('pgf/boxplots.pgf')
