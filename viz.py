import matplotlib.pyplot as plt
from clyent import color


def forward_curve(prices):
    plt.plot(prices[:, 0], color='tab:blue', linewidth=3)
    for t in range(prices.shape[0]):
        plt.plot(range(t, t + prices.shape[1]), prices[t], marker='o', color='tab:orange', ls='--', markersize=10)
    plt.show()

def decision_curve(prices, decisions):
    plt.plot(prices[:, 0], color='tab:blue', linewidth=3)
    for t in range(prices.shape[0]):
        plt.plot(range(t, min(t + prices.shape[1], prices.shape[0])),
                 prices[t, :min(prices.shape[1], prices.shape[0] - t)],
                 marker='o', color='tab:orange', ls='--', markersize=10)
        for tau in range(prices.shape[1]):
            if decisions[t, tau]:
                plt.plot(t + tau, prices[t, tau], marker='o', fillstyle='none', color='tab:blue', markersize=20)
    plt.show()
