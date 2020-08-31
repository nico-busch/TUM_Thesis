import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

np.random.seed(42000)

spot = np.empty(36)
positive = False
while not positive:
    spot[0] = 200
    for t in range(36 - 1):
        spot[t + 1] = spot[t] + np.random.normal(0, 10)
    if np.all(spot >= 0):
        positive = True

plt.figure(figsize=(14 / 2.54, 3))
ax = plt.gca()

ax.plot(spot, marker='o', color=(0, 101 / 255, 189 / 255), linewidth=1, markersize=5)
input_window = plt.Rectangle((6, spot.min() + 2), 12, spot.max() - spot.min() + 2,
                             color='lightgrey')
ax.add_patch(input_window)
output_window = plt.Rectangle((19, spot.min() + 2), 6, spot.max() - spot.min() + 2,
                             color='lightgrey')
ax.add_patch(output_window)

plt.ylim(spot.min() - 2, spot.max() + 8)
ax.text(12, spot.min() + 5, 'Input window', horizontalalignment='center')
ax.text(22, spot.min() + 5, 'Output' + '\n' + 'window', horizontalalignment='center')

ax.autoscale(axis='x', tight=True)

# plt.show()
plt.savefig('misc/windows.pgf')