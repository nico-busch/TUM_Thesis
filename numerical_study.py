import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import product

import test_utils
import dda
import prescriptive
import predictive
import viz

def generate_data(n_time=100, initial_spot=200, sigma=5, beta0=0, beta1=1, beta2=1, seasonal=True, mean_demand=1,
                  n_add_feature=8, price_feature_sigma=15):

    spot = np.empty(n_time)
    price_feature = np.empty(n_time)

    positive = False
    while not positive:
        price_feature[:] = np.random.normal(0, price_feature_sigma, n_time)
        spot[0] = initial_spot
        for t in range(n_time - 1):
            spot[t + 1] = beta0 + beta1 * spot[t] + beta2 * price_feature[t] + np.random.normal(0, sigma)
        if np.all(spot >= 0):
            positive = True

    forward = spot + np.random.normal(0, spot / 100)
    prices = np.vstack([spot, forward]).T

    add_features = np.hstack([np.random.normal(10 * i, 2 * i, [n_time, 1]) for i in range(3, 3 + n_add_feature)])
    features = np.hstack([spot[:, np.newaxis], price_feature[:, np.newaxis], add_features])

    if seasonal:
        demand = 1 + 0.5 * np.sin(np.pi * (np.arange(1, n_time + 1) - 2) / 6) * mean_demand
    else:
        demand = np.ones(n_time)

    return prices, features, demand


def simulation():

    np.random.seed(42)
    torch.manual_seed(42)

    n_runs = 100

    betas = [
        {
            'beta0': 0,
            'beta1': 1
        },
        {
            'beta0': 100,
            'beta1': 0.5}
    ]
    sigmas = [5, 10, 20, 30]
    train_sizes = [24, 48, 72]
    test_size = 48

    results = pd.DataFrame(columns=['beta0', 'beta1', 'sigma', 'train_size', 'run',
                                    'p_spot', 'p_m1', 'dda_ml1', 'dda_ml2'])
    results = results.set_index(['beta0', 'beta1', 'sigma', 'train_size', 'run'])

    for beta, sigma, train_size in product(betas, sigmas, train_sizes):

        for r in range(n_runs):

            prices, features, demand = generate_data(n_time=train_size + test_size - 1,
                                                     beta0=beta['beta0'], beta1=beta['beta1'], sigma=sigma)

            if beta['beta0'] == 0:
                continue

            c_pf = test_utils.c_pf(prices[-test_size:], demand[-test_size:])
            c_0 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 0)
            c_1 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 1)
            c_dda1 = test_dda(prices, features, demand, test_size, 'l1')
            c_dda2 = test_dda(prices, features, demand, test_size, 'l2')
            # c_pred = test_predictive(prices, features, demand, test_size)
            # c_pres = test_prescriptive(prices, features, demand, test_size)

            costs = np.array([c_0, c_1, c_dda1, c_dda2])
            pe = (costs - c_pf) / c_pf * 100

            results.loc[beta['beta0'], beta['beta1'], sigma, train_size, r + 1] = pe
            print(results)

            results.to_pickle('data/numerical_dda3.pkl')

def test_prescriptive(prices, features, demand, test_size):

    params = {
        'n_steps': 1,
        'n_hidden': 128,
        'n_layers': 4,
        'batch_size': 32,
        'n_epochs': 25,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.0
    }

    scaler = StandardScaler()
    scaler.fit(features[:-test_size])
    features_std = scaler.transform(features)

    model = prescriptive.model.PrescriptiveNet(prices.shape[1], features.shape[1], params['n_steps'],
                                               params['n_hidden'], params['n_layers'], params['dropout'])
    train_set = prescriptive.dataset.Dataset(prices[:-test_size + 1], features_std[:-test_size + 1],
                                             demand[:-test_size + 1], params['n_steps'])
    trainer = prescriptive.train.Trainer(model, train_set, train_set, params)
    trainer.train()

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)
    for t in range(test_size):
        idx = prices.shape[0] - test_size + t
        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(features_std[None, idx - params['n_steps'] + 1:idx + 1]).float())
            signals[t, 1:] = probs.sigmoid().numpy().squeeze(axis=0) >= 0.5

    costs, _ = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    return costs.mean()

def test_predictive(prices, features, demand, test_size):

    params = {
        'n_steps': 1,
        'n_hidden': 128,
        'n_layers': 4,
        'batch_size': 32,
        'n_epochs': 25,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.0
    }

    scaler = StandardScaler()
    scaler.fit(features[:-test_size])
    features_std = scaler.transform(features)

    model = predictive.model.PredictiveNet(features.shape[1], params['n_steps'], params['n_hidden'],
                                           params['n_layers'], params['dropout'])
    train_set = predictive.dataset.Dataset(features_std[:-test_size + 1], params['n_steps'])
    trainer = predictive.train.Trainer(model, train_set, train_set, params)
    trainer.train()

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)
    for t in range(test_size):
        idx = prices.shape[0] - test_size + t
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(features_std[None, idx - model.n_steps + 1:idx + 1]).float())
            preds = preds * scaler.scale_[0] + scaler.mean_[0]
            signals[t, 1:] = preds >= prices[idx, 0]

    costs, _ = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    return costs.mean()

def test_dda(prices, features, demand, test_size, reg):

    features = np.hstack([np.ones([features.shape[0], 1]), features])

    scaler = StandardScaler()
    scaler.fit(features[:-test_size])
    features_std = scaler.transform(features)

    prices_train, features_train, demand_train = prices[:-test_size], features_std[:-test_size], demand[:-test_size]
    prices_test, features_test, demand_test = prices[-test_size:], features_std[-test_size:], demand[-test_size:]

    model = dda.model.DDA(prices_train, features_train, demand_train, reg=reg, big_m=False)
    model.train()

    costs = model.prescribe(prices_test, features_test, demand_test, model.beta)

    return costs.mean()

if __name__ == '__main__':
    simulation()






