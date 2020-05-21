import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def c_pf(prices, demand):

    import gurobipy as gp
    from gurobipy import GRB

    T = prices.shape[0]
    F = prices.shape[1] - 1

    p = {(t, tau): prices[t - 1, tau] for t in range(1, T + 1) for tau in range(F + 1)}
    d = {t: demand[t - 1] for t in range(1, T + 1)}

    m = gp.Model()
    m.Params.outputFlag = 0
    q = m.addVars(range(1, T + 1), range(F + 1))

    m.setObjective(gp.quicksum(p[t, tau] * q[t, tau]
                               for t in range(1, T + 1) for tau in range(F + 1) if tau <= T - t) / T, GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(q[t - tau, tau] for tau in range(F + 1) if tau <= t - 1) == d[t]
                 for t in range(1, T + 1))

    m.optimize()

    return m.objVal

def c_tau(prices, demand, tau):

    if tau == 0:
        costs = prices[:, 0] * demand

    else:
        costs = np.zeros(prices.shape[0])
        costs[:-tau] += prices[:-tau, tau] * demand[tau:]
        costs[:tau] += prices[:tau, 0] * demand[:tau]

    return costs.mean()

def test_prescriptive(prices, features, demand, test_size):

    from prescriptive.model import RNN
    from prescriptive.dataset import Dataset
    from prescriptive.train import Trainer

    params = {
        'n_steps': 1,
        'n_hidden': 100,
        'n_layers': 4,
        'batch_size': 4,
        'n_epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.0
    }

    scaler = StandardScaler()
    scaler.fit(features[:-test_size])
    features_std = scaler.transform(features)

    model = RNN(prices.shape[1], features.shape[1],
                params['n_steps'], params['n_hidden'], params['n_layers'], params['dropout'])

    prices_train, features_train, demand_train = prices[:-test_size], features_std[:-test_size], demand[:-test_size]
    prices_test, features_test, demand_test = prices[-test_size:], features_std[-test_size:], demand[-test_size:]
    train_set = Dataset(prices_train, features_train, demand_train, params['n_steps'])
    test_set = Dataset(prices_test, features_test, demand_test, params['n_steps'])
    trainer = Trainer(model, train_set, test_set, params)
    trainer.train()

    costs = np.zeros(prices_test.shape[0])
    satisfied = np.zeros(prices_test.shape[0], dtype=np.bool_)

    for t in range(prices_test.shape[0]):

        model.eval()
        with torch.no_grad():

            if not satisfied[t]:
                costs[t] += prices_test[t, 0] * demand_test[t]
                satisfied[t] = True

            if params['n_steps'] - 1 <= t <= prices_test.shape[0] - prices_test.shape[1]:

                probs = model(torch.tensor(features_test[None, t - params['n_steps'] + 1:t + 1]).float())
                signals = probs.sigmoid().numpy().squeeze(axis=0) >= 0.5

                for tau in range(1, prices_test.shape[1]):

                    if not satisfied[t + tau] and signals[tau - 1]:
                        costs[t] += prices_test[t, tau] * demand_test[t + tau]
                        satisfied[t + tau] = True

    return costs.mean()

def test_predictive(prices, features, demand, test_size):

    from predictive.model import RNN
    from predictive.dataset import Dataset
    from predictive.train import Trainer

    params = {
        'n_steps': 1,
        'n_hidden': 100,
        'n_layers': 4,
        'batch_size': 4,
        'n_epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.5
    }

    scaler = StandardScaler()
    scaler.fit(features[:-test_size])
    features_std = scaler.transform(features)

    model = RNN(features.shape[1], params['n_steps'], params['n_hidden'], params['n_layers'], params['dropout'])
    prices_train, features_train, demand_train = prices[:-test_size], features_std[:-test_size], demand[:-test_size]
    prices_test, features_test, demand_test = prices[-test_size:], features_std[-test_size:], demand[-test_size:]
    train_set = Dataset(features_train, params['n_steps'])
    test_set = Dataset(features_train, params['n_steps'])
    trainer = Trainer(model, train_set, test_set, params)
    trainer.train()

    costs = np.zeros(prices_test.shape[0])
    satisfied = np.zeros(prices_test.shape[0], dtype=np.bool_)

    for t in range(prices_test.shape[0]):

        model.eval()
        with torch.no_grad():

            if not satisfied[t]:
                costs[t] += prices_test[t, 0] * demand_test[t]
                satisfied[t] = True

            if model.n_steps - 1 <= t <= prices_test.shape[0] - prices_test.shape[1]:

                preds = model(torch.tensor(features[None, t - model.n_steps + 1:t + 1]).float())
                preds = preds * scaler.scale_[0] + scaler.mean_[0]

                for tau in range(1, prices_test.shape[1]):

                    if not satisfied[t + tau] and preds[tau - 1] >= prices_test[t, tau]:
                        costs[t] += prices_test[t, tau] * demand_test[t + tau]
                        satisfied[t + tau] = True

    return costs.mean()

def test_dda(prices, features, demand, test_size, reg):

    from dda.model import DDA

    features = np.hstack([np.ones([features.shape[0], 1]), features])

    scaler = StandardScaler()
    scaler.fit(features[:-test_size])
    features_std = scaler.transform(features)

    prices_train, features_train, demand_train = prices[:-test_size], features_std[:-test_size], demand[:-test_size]
    prices_test, features_test, demand_test = prices[-test_size:], features_std[-test_size:], demand[-test_size:]

    dda = DDA(prices_train, features_train, demand_train, reg=reg, big_m=False)
    dda.train()

    costs = dda.prescribe(prices_test, features_test, demand_test, dda.beta)

    return costs.mean()
