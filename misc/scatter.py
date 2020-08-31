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

df = pd.read_csv('data/reuters.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)

fig, axes = plt.subplots(figsize=(14/2.54, 3.1), ncols=4, nrows=2, sharey=True, sharex=True)

futures = ['M1', 'M2', 'M3', 'M4']

for i, row in enumerate(axes):
    for j, (ax, fut) in enumerate(zip(row, futures)):
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', lw=1, zorder=1)
        if i == 0:
            ax.scatter(df[fut], df['SPOT'], color=(0, 101/255, 189/255), alpha=0.25, lw=0, zorder=2)
        else:
            ax.scatter(df[fut], df['SPOT'].shift(-j), color=(0, 101 / 255, 189 / 255), alpha=0.25, lw=0, zorder=2)
            ax.set_xlabel(fut + '($t$)')
            ax.annotate('$t^{\prime}=' + str(j + 1) +'$', xy=(1, 0), xycoords='axes fraction', xytext=(-3, 3),
                        textcoords='offset points', ha='right', va='bottom')
        ax.axis('square')

axes[0][0].set_ylabel('SPOT($t$)')
axes[1][0].set_ylabel('SPOT($t+t^{\prime}$)')
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# plt.show()
plt.savefig('misc/scatter.pgf')