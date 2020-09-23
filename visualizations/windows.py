import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

tum1 = (0, 101/255, 189/255)
tum2 = (0, 101/255, 189/255, 0.5)
tum3 = (227/255, 114/255, 34/255, 0.5)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Simulate RW time series
np.random.seed(42000)
spot = np.empty(36)
positive = False
while not positive:
    spot[0] = 200
    for t in range(36 - 1):
        spot[t + 1] = spot[t] + np.random.normal(0, 10)
    if np.all(spot >= 0):
        positive = True

# Plot time windows
plt.figure(figsize=(14 / 2.54, 3))
ax = plt.gca()
ax.plot(range(1, 37), spot, marker='o', color=tum1, linewidth=1, markersize=5)
input_window = plt.Rectangle((7, spot.min() - 8), 11, spot.max() - spot.min() + 16,
                             color=tum2, linewidth=0, zorder=3, label='Input window')
ax.add_patch(input_window)
output_window = plt.Rectangle((19, spot.min() - 8), 5, spot.max() - spot.min() + 16,
                              color=tum3, linewidth=0, zorder=3, label='Output window')
ax.add_patch(output_window)
plt.ylim(spot.min() - 8, spot.max() + 8)
ax.arrow(7, spot.max() + 12, 17, 0, length_includes_head=True, head_width=2, head_length=0.5,
         color=(88/255, 88/255, 90/255), zorder=4, clip_on=False)

ax.autoscale(axis='x', tight=True)
plt.xlabel('$t$')
plt.legend()
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
plt.savefig('pgf/windows.pgf')