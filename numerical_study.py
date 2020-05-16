import torch
import numpy as np
import pandas as pd
from itertools import product

from pa_rnn.train import Trainer
from dda.dda import DDA

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
    forward2 = forward + np.random.normal(0, forward / 100)
    prices = np.vstack([spot, forward]).T

    add_features = np.hstack([np.random.normal(10 * i, 2 * i, [n_time, 1]) for i in range(3, 3 + n_add_feature)])
    features = np.hstack([spot[:, np.newaxis], forward[:, np.newaxis], price_feature[:, np.newaxis], add_features])

    if seasonal:
        demand = 1 + 0.5 * np.sin(np.pi * (np.arange(1, n_time + 1) - 2) / 6) * mean_demand
    else:
        demand = np.ones(n_time)

    return prices, features, demand


def simulation(n_runs=100):

    np.random.seed(3)
    torch.manual_seed(3)

    beta = [
        {
            'beta0': 0,
            'beta1': 1
        },
        {
            'beta0': 100,
            'beta1': 0.5}
    ]
    sigma = [5, 10, 20, 30]
    train_size = [24]
    test_size = 48

    params = {
        'n_steps': 12,
        'n_hidden': 100,
        'n_layers': 4,
        'batch_size': 4,
        'n_epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.0
    }

    results = pd.DataFrame(columns=['Beta0', 'Beta1', 'Sigma', 'Train', 'Run',
                                    'P-Spot', 'P-M1',  'PA_RNN'])
    results = results.set_index(['Beta0', 'Beta1', 'Sigma', 'Train', 'Run'])

    for b, s, t in product(beta, sigma, train_size):

        for r in range(n_runs):

            prices, features, demand = generate_data(n_time=t + test_size,
                                                     beta0=b['beta0'], beta1=b['beta1'], sigma=s)

            prices_train, features_train, demand_train = prices[:t], features[:t], demand[:t]
            prices_test, features_test, demand_test = prices[t:], features[t:], demand[t:]

            c_pf = DDA.c_pf(prices_test, demand_test)

            c_spot = np.mean(prices_test[:, 0] * demand_test)
            c_forward = np.mean(np.hstack([prices_test[0, 0], prices_test[:-1, 1]]) * demand_test)

            # dda_l1 = DDA(prices_train, features_train, demand_train, prices_test, features_test, demand_test, reg='l1')
            # dda_l1.train()
            # c_dda_l1 = dda_l1.test().mean()
            #
            # dda_l2 = DDA(prices_train, features_train, demand_train, prices_test, features_test, demand_test, reg='l2')
            # dda_l2.train()
            # c_dda_l2 = dda_l2.test().mean()

            trainer = Trainer(prices, features, demand, test_size, params)
            trainer.train()
            c_nn = trainer.test().mean()

            costs = np.array([c_spot, c_forward, c_nn])
            pe = (costs - c_pf) / c_pf * 100

            results.loc[b['beta0'], b['beta1'], s, t, r + 1] = pe
            print(results)

            results.to_pickle('data/results.pkl')






