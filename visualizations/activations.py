import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Sigma, tanh and relu
x = np.linspace(-10, 10, 1000)
y1 = np.e ** x / (np.e ** x + 1)
y2 = (np.e ** (2 * x) - 1) / (np.e ** (2 * x) + 1)
y3 = np.maximum(0, x)

# Plot curves
plt.figure(figsize=(4, 4 / 16 * 9))
ax = plt.gca()
ax.plot(x, y1, color=(0, 101/255, 189/255), linewidth=1, label='$\sigma$')
ax.plot(x, y2, color=(227/255, 114/255, 34/255), linewidth=1, label=r'$\tanh$')
ax.plot(x, y3, color=(162/255, 173/255, 0), linewidth=1, label=r'$\mathrm{ReLU}$', zorder=3)
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.set_ylim(-1.02, 1)
ax.set_xlim(-6, 6.02)
ax.grid(True, which='both', ls='-')
plt.legend()
plt.locator_params(axis='y', nbins=5)
plt.tight_layout()
plt.show()
plt.savefig('visualizations/activations.pgf')
