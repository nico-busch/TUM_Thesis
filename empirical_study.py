import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import prescriptive
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
    features = df[['SPOT', 'SPOT-1', 'SPOT-2', 'SPOT-3', 'GASPOOL',
       'TTF', 'HENRYHUB', 'API2', 'BRENT', 'BCOM', 'EURUSD', 'EURGBP',
       'EURCNY', 'DXY', 'PPI', 'PROD', 'CONS', 'FUNDRATE', 'SP500', 'TEMP']].to_numpy()
    demand = np.ones(prices.shape[0])

    viz.forward_curve(prices[:12])

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
    c_nn = test_prescriptive(prices, features, demand, test_size)

    costs = np.array([c_0, c_1, c_2, c_3, c_4, c_rand, c_nn])
    pe = (costs - c_pf) / c_pf * 100

    print(pe)

def test_prescriptive(prices, features, demand, test_size):

    params = {
        'n_steps': 3,
        'n_hidden': 128,
        'n_layers': 4,
        'batch_size': 32,
        'n_epochs': 25,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.0
    }

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)

    for t in range(test_size):

        idx = prices.shape[0] - test_size + t

        scaler = StandardScaler()
        scaler.fit(features[:idx + 1])
        features_std = scaler.transform(features)

        model = prescriptive.model.PrescriptiveNet(prices.shape[1], features.shape[1], params['n_steps'],
                                                   params['n_hidden'], params['n_layers'], params['dropout'])
        train_set = prescriptive.dataset.PresDataset(prices[:idx + 1], features_std[:idx + 1], demand[:idx + 1],
                                                     params['n_steps'])
        val_set = prescriptive.dataset.PresDataset(prices[idx - params['n_steps'] + 1:idx + 16],
                                               features_std[idx - params['n_steps'] + 1:idx + 16],
                                               demand[idx - params['n_steps'] + 1:idx + 16],
                                                   params['n_steps'])
        trainer = prescriptive.train.PresTrainer(model, train_set, val_set, params)
        trainer.train()

        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(features_std[None, idx - params['n_steps'] + 1:idx + 1]).float())
            signals[t, 1:] = probs.sigmoid().numpy().squeeze(axis=0) >= 0.5

    costs, decisions = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)
    viz.decision_curve(prices[-test_size:], decisions)

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


