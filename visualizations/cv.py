import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from sklearn.model_selection import TimeSeriesSplit

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Simulate time series split
time = list(range(1, 51))
n_splits = 3
cv = TimeSeriesSplit(n_splits, len(time))

# Plot cross-validation
cmap = matplotlib.colors.ListedColormap([(0, 101/255, 189/255), (100/255, 160/255, 200/255)])
plt.figure(figsize=(4, 4 / 16 * 9))
ax = plt.gca()
for i, (train, val) in enumerate(cv.split(time)):
    idx = np.array([np.nan] * len(time))
    idx[train] = 0
    idx[val] = 1
    ax.scatter(time, [i + 0.5] * len(time),
               c=idx, marker='_', lw=20, cmap=cmap)

    yticklabels = list(range(1, n_splits + 1))
    ax.set(yticks=np.arange(n_splits) + .5, yticklabels=yticklabels,
           xlabel='$t$', ylabel="Iteration",
           ylim=[n_splits + 0.2, -.2], xlim=[1, len(time)])

    ax.legend([Patch(color=cmap(0)), Patch(color=cmap(1))], ['Training set', 'Validation set'])
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.show()
plt.savefig('pgf/cv.pgf')