import numpy as np

from model import PRNN
from dataset import PRNNDataset
from train import PRNNTrainer


def generate_data(n_time=96, initial_spot=200, sigma=5, beta0=0, beta1=1, beta2=1, seasonal=True, mean_demand=1,
                  n_add_feature=7, price_feature_sigma=15):

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

    add_feature = np.hstack([np.random.normal(10 * i, 2 * i, [n_time, 1]) for i in range(3, 3 + n_add_feature)])
    covariates = np.hstack([price_feature[:, np.newaxis], add_feature])

    if seasonal:
        demand = 1 + 0.5 * np.sin(np.pi * (np.arange(1, n_time + 1) - 2) / 6) * mean_demand
    else:
        demand = np.ones(n_time)

    return prices, covariates, demand


def simulation():

    n_hidden = 100
    n_steps = 6
    n_runs = 10
    train_size = 48

    for r in range(n_runs):

        prices, covariates, demand = generate_data()

        n_prices = prices.shape[1]
        n_features = n_prices + covariates.shape[1]

        model = PRNN(n_steps, n_features, n_prices, n_hidden)

        prices_train, covariates_train, demand_train = prices[:train_size], covariates[:train_size], demand[:train_size]
        train_set = PRNNDataset(prices_train, covariates_train, demand_train, n_steps)
        prices_test, covariates_test, demand_test = prices[train_size:], covariates[train_size:], demand[train_size:]
        val_set = PRNNDataset(prices_test, covariates_test, demand_test, n_steps)
        trainer = PRNNTrainer(model, train_set, val_set)


        trainer.train()

        exit()




