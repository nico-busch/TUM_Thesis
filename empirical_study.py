import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import presnet
import dda
import test_utils
import viz


def run():

    np.random.seed(42)
    torch.manual_seed(42)

    df = pd.read_csv('data/data.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)
    df = df[12:]
    df = df.bfill()
    prices = df[['SPOT', 'M1', 'M2', 'M3', 'M4']].to_numpy()
    features = df[['SPOT', 'EURUSD', 'TEMP']].to_numpy()
    demand = np.ones(prices.shape[0])

    # viz.forward_curve(prices[:12])

    test_size = 36

    c_0 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 0)
    c_1 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 1)
    c_2 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 2)
    c_3 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 3)
    c_4 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 4)

    c_rand = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:],
                                    np.vstack([np.random.choice(2, test_size, p=[1 - (1 / p), (1 / p)])
                                               for p in range(1, prices.shape[1] + 1)]).T)[0].mean()
    c_pf = test_utils.c_pf(prices[-test_size:], demand[-test_size:])

    # c_dda = test_dda(prices, features, demand, test_size, reg='l1')
    c_nn = test_presnet(prices, features, demand, test_size, 'lstm')

    costs = np.array([c_0, c_1, c_2, c_3, c_4, c_rand, c_nn])
    pe = (costs - c_pf) / c_pf * 100

    print(pe)


def test_presnet(prices, features, demand, test_size, cell_type, n_ensemble=10):

    params = {
        'n_steps': 12,
        'n_hidden': 128,
        'n_layers': 4,
        'batch_size': 32,
        'n_epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.1
    }

    scores = np.zeros([test_size, prices.shape[1] - 1])
    for i in range(n_ensemble):

        for t in range(test_size):

            idx = prices.shape[0] - test_size + t
            model = presnet.model.PresNet(cell_type, prices.shape[1], features.shape[1], params['n_steps'],
                                          params['n_hidden'], params['n_layers'], params['dropout'])
            train_set = presnet.dataset.PresDataset(prices[:idx + 1], features[:idx + 1], demand[:idx + 1],
                                                    params['n_steps'])
            trainer = presnet.train.PresTrainer(model, train_set, params)
            trainer.train()

            model.eval()
            with torch.no_grad():
                sequence = train_set.scaler.transform(features)[None, idx - params['n_steps'] + 1:idx + 1]
                probs = model(torch.tensor(sequence).float())
                scores[t] += probs.sigmoid().numpy().squeeze(axis=0)

    scores = scores / n_ensemble
    signals = np.hstack([np.zeros([test_size, 1]), scores])

    viz.roc(prices[-test_size:], scores)
    costs, decisions = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    # viz.decision_curve(prices[-test_size:], decisions)

    return costs.mean()


def test_dda(prices, features, demand, test_size, reg):
    features = np.hstack([np.ones([features.shape[0], 1]), features])
    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)

    for t in range(test_size):
        idx = prices.shape[0] - test_size + t

        scaler = StandardScaler()
        scaler.fit(features[:idx + 1])
        features_std = scaler.transform(features)

        model = dda.model.DDA(prices[:idx + 1], features_std[:idx + 1], demand[:idx + 1], reg=reg, big_m=False)
        model.train()

        thresholds = np.sum(features_std[idx, :, None] * model.beta, axis=0)
        signals[t] = (prices[idx] <= thresholds)

    costs, decisions = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)
    viz.decision_curve(prices[-test_size:], decisions)

    return costs.mean()


if __name__ == '__main__':
    run()
