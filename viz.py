import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def forward_curve(prices):
    plt.plot(prices[:, 0], color='tab:blue', linewidth=3)
    for t in range(prices.shape[0]):
        plt.plot(range(t, t + prices.shape[1]), prices[t], marker='o', color='tab:orange', ls='--', markersize=10)
    plt.savefig('example.pgf')
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

def roc(prices, scores):
    targets = []
    for t in range(prices.shape[0] - prices.shape[1] + 1):
        options = [np.flipud(np.flipud(prices[t:t + i + 1]).diagonal()) for i in range(1, prices.shape[1])]
        targets.append(np.array([int(x.argmin(axis=0) == 0) for x in options]))
    targets = np.vstack(targets)
    scores = scores[:-prices.shape[1] + 1]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    colors = [(0, 101/255, 189/255), (227/255, 114/255, 34/255), (162/255, 173/255, 0), (100/255, 160/255, 200/255)]

    for tau in range(prices.shape[1] - 1):
        fpr, tpr, thresholds = roc_curve(targets[:, tau], scores[:, tau])
        ax.plot(fpr, tpr, label='M' + str(tau + 1), color=colors[tau])

    ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='black')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    ax.set_aspect('equal', 'box')

    plt.legend()
    plt.show()

