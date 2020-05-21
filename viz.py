import matplotlib.pyplot as plt
from clyent import color


def forward_curve(prices):

    plt.plot(prices[:, 0], color='tab:blue', linewidth=3)
    for t in range(prices.shape[0]):
        plt.plot(range(t, t + prices.shape[1]), prices[t], marker='o', color='tab:orange', ls='--', markersize=10)
    plt.show()
