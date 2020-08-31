import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import shap
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

tum1 = (0, 101 / 255, 189 / 255)
tum2 = (227/255, 114/255, 34/255)

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     'pgf.texsystem': 'pdflatex',
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })


def forward_curve(prices, dates=None):

    # Plot the forward curve including spot and futures prices
    plt.figure(figsize=(14 / 2.54, 3))
    ax = plt.gca()
    if dates is None:
        plt.plot(prices[:, 0], color=tum1, linewidth=3, label='Spot')
    else:
        plt.plot(dates, prices[:, 0], color=tum1, linewidth=3, label='SPOT')
    for t in range(prices.shape[0]):
        if dates is None:
            plt.plot(range(t, min(t + prices.shape[1], prices.shape[0])),
                     prices[t, :min(prices.shape[1], prices.shape[0] - t)],
                     marker='o', color=tum2, ls='-', markersize=3, label='Forward curve' if t == 0 else '')
        else:
            plt.plot([dates.min() + pd.DateOffset(months=x) for x in range(t, min(t + prices.shape[1], prices.shape[0]))],
                     prices[t, :min(prices.shape[1], prices.shape[0] - t)],
                     marker='o', color=tum2, ls='-', markersize=3, label='M1, M2, M3, M4' if t == 0 else '')

    ax.yaxis.grid()

    # Support for dates
    if dates is not None:
        ax.autoscale(axis='x', tight=True)
        plt.ylabel('€/MWh')
        plt.xticks(rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.2)

    plt.legend()
    plt.savefig('misc/forward_curve.pgf')
    # plt.show()


def decision_curve(prices, decisions, dates=None):

    # Plot the forward curve including spot and futures prices
    plt.figure(figsize=(14 / 2.54, 3))
    ax = plt.gca()

    plt.plot(dates, prices[:, 0], color=tum1, linewidth=3, label='Spot')
    for t in range(prices.shape[0]):
        if dates is None:
            plt.plot(range(t, min(t + prices.shape[1], prices.shape[0])),
                     prices[t, :min(prices.shape[1], prices.shape[0] - t)],
                     marker='o', color=tum2, ls='-', markersize=3, label='Forward curve' if t == 0 else '')
        else:
            plt.plot([dates.min() + pd.DateOffset(months=x) for x in range(t, min(t + prices.shape[1], prices.shape[0]))],
                     prices[t, :min(prices.shape[1], prices.shape[0] - t)],
                     marker='o', color=tum2, ls='-', markersize=3, label='Forward curve' if t == 0 else '')

        # Plot decisions
        for tau in range(prices.shape[1]):
            if decisions[t, tau]:
                if dates is None:
                    plt.plot(t + tau, prices[t, tau], marker='o', fillstyle='none', color=tum1, markersize=20,
                             label='Decision' if t == 0 and tau == 0 else '', linestyle='None')
                else:
                    plt.plot(dates.min() + pd.DateOffset(months=t + tau), prices[t, tau], marker='o', fillstyle='none',
                             color=tum1, markersize=10, label='Decision' if t == 0 and tau == 0 else '', linestyle='None')

    ax.yaxis.grid()

    # Support for dates
    if dates is not None:
        ax.autoscale(axis='x', tight=True)
        plt.ylabel('€/MWh')
        plt.xticks(rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.2)

    plt.legend()
    # plt.savefig('misc/forward_curve.pgf')
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


def feature_importance(model, sequences, cols):

    explainer = shap.DeepExplainer(model, sequences)
    shap_values = explainer.shap_values(sequences)
    shap.summary_plot(shap_values[0].reshape(-1, sequences.shape[2]),
                    features=sequences.contiguous().view(-1, sequences.shape[2]).numpy(),
                    max_display=100, feature_names=cols)


def hedge(df, decisions):

    plt.figure(figsize=(14 / 2.54, 3))
    ax = plt.gca()

    plt.fill_between(df.index, df['Demand'], step='mid', color=tum1, label='Unhedged')

    hedged = np.zeros(decisions.shape[0])
    for t in range(decisions.shape[0]):
        for tau in range(1, decisions.shape[1]):
            if decisions[t, tau] == 1:
                hedged[t + tau] = 1

    hedged = pd.DataFrame(data={'Signal': hedged, 'Month': df['Month'].unique()})
    df = pd.merge(df, hedged, on='Month').set_index(df.index)
    df['Signal'] = df['Signal'].astype('int64')

    plt.fill_between(df.index, df['Level'] * df['Signal'], step='mid', color=tum2, label='Hedged')

    ax.autoscale(axis='x', tight=True)
    plt.ylim(bottom=0)
    plt.ylabel('MW')
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.2)
    plt.legend()

    plt.show()
