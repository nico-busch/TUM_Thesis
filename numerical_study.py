import numpy as np

from train import Trainer
from model import NN


def generate_data(n_time=96, initial=200, sigma=5, beta0=0, beta1=1, beta2=1):

    price_feature = np.random.normal(0, 15, n_time)
    add_feature = np.vstack([np.random.normal(10 * i, (2 * i) ** 2, n_time) for i in range(2, 10)]).T
    spot = []
    for t in range(n_time):
        if t == 0:
            spot.append(beta0 + beta1 * initial + beta2 * price_feature[t] + np.random.normal(0, sigma))
        else:
            spot.append(beta0 + beta1 * spot[-1] + beta2 * price_feature[t] + np.random.normal(0, sigma))
    spot = np.array(spot)
    forward = (spot + np.random.normal(0, (spot / 100) ** 2))[:, np.newaxis]
    covariate = np.hstack([price_feature[:, np.newaxis], add_feature])
    demand = np.ones(n_time)

    return spot, forward, covariate, demand


def simulation():

    n_hidden = 100
    n_steps = 12
    n_forwards = 1
    n_input = 11
    split = 0.5
    n_runs = 100

    results = []
    for x in range(n_runs):
        model = NN(n_steps, n_input, n_forwards, n_hidden)
        spot, forward, covariate, demand = generate_data()
        trainer = Trainer(model, spot, forward, covariate, demand, split)
        trainer.train()
        results.append(trainer.test().detach().numpy())

    print(np.mean(results))




