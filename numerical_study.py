import torch
import numpy as np
import pandas as pd
from itertools import product

import tests

def generate_data(n_time=120, initial_spot=200, sigma=5, beta0=0, beta1=1, beta2=1, seasonal=True, mean_demand=1,
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
    sigmas = [5]
    train_sizes = [72]
    test_size = 48

    results = pd.DataFrame(columns=['beta0', 'beta1', 'sigma', '#train', 'run',
                                    'p_spot', 'p_m1', 'rnn'])
    results = results.set_index(['beta0', 'beta1', 'sigma', '#train', 'run'])

    for beta, sigma, train_size in product(betas, sigmas, train_sizes):

        for r in range(n_runs):

            prices, features, demand = generate_data(n_time=train_size + test_size,
                                                     beta0=beta['beta0'], beta1=beta['beta1'], sigma=sigma)

            c_pf = tests.c_pf(prices[train_size:], demand[train_size:])
            c_0 = tests.c_tau(prices[train_size:], demand[train_size:], 0)
            c_1 = tests.c_tau(prices[train_size:], demand[train_size:], 1)
            # c_dda1 = test.test_dda(prices, features, demand, test_size, 'l1')
            # c_dda2 = test.test_dda(prices, features, demand, test_size, 'l2')
            # c_pred = test.test_predictive(prices, features, demand, test_size)
            c_pres = tests.test_prescriptive(prices, features, demand, test_size)

            costs = np.array([c_0, c_1, c_pres])
            pe = (costs - c_pf) / c_pf * 100

            results.loc[beta['beta0'], beta['beta1'], sigma, train_size, r + 1] = pe
            results.loc[beta['beta0'], beta['beta1'], sigma, train_size, 'Mean'] = results.mean(axis=0)
            print(results)

            results.to_pickle('data/results.pkl')

if __name__ == '__main__':
    simulation()






