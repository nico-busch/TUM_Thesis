import numpy as np
import matplotlib.pyplot as plt

tum2 = (227/255, 114/255, 34/255)

x = [1, 2, 3, 4, 5]
y = [1, 4, 7, 9, 10]

plt.plot(x, y,  marker='o', color=tum2, ls='-', markersize=20, linewidth=10)
plt.xlim(0, 6)
plt.ylim(0, 12)
plt.show()

