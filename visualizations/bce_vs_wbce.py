import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

tum1 = (0, 101/255, 189/255)
tum2 = (227/255, 114/255, 34/255)
tum3 = (162/255, 173/255, 0)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Prepare dataframes
wbce = pd.read_pickle('results/numerical_study.pkl')
wbce = wbce[['mlp_ieo', 'rnn_ieo', 'lstm_ieo']]
wbce.columns = ['MLP-IEO', 'RNN-IEO', 'LSTM-IEO']
bce = pd.read_pickle('results/numerical_study_bce.pkl')
bce.columns = ['MLP-IEO', 'RNN-IEO', 'LSTM-IEO']
bce = bce.loc[pd.IndexSlice[:, :, 10, :, 73], :]
wbce = wbce.loc[pd.IndexSlice[:, :, 10, :, 73], :]

# Plot scatter plots
fig, axes = plt.subplots(figsize=(14/2.54, 2.5), ncols=3, sharey=True, sharex=True)
col = [tum1, tum2, tum3]
for i, (ax, alg) in enumerate(zip(axes, wbce.columns)):
    ax.scatter(wbce[alg], bce[alg], alpha=0.25, lw=0, label=alg, color=col[i])
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', lw=1, zorder=1)
    ax.axis('square')
    ax.set_xlabel('Weighted BCE')
    if i == 0:
        ax.set_ylabel('BCE')
    ax.set_title(alg, fontdict={'fontsize': 10})
    loc = plticker.MultipleLocator(2)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
plt.ylim(bottom=0, top=5)
plt.xlim(left=0, right=5)
plt.tight_layout()
plt.show()
plt.savefig('pgf/wbce.pgf')