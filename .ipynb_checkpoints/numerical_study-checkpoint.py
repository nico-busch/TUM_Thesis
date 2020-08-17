import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

import test_utils
import dda
import presnet
import prednet
import viz

def generate_series(n_time=100, initial_spot=200, sigma=5, beta0=0, beta1=1, beta2=1, seasonal=False,
                    n_add_feature=8, feature_sigma=15, seasonal_demand=True, mean_demand=1):

    spot = np.empty(n_time)
    price_feature = np.empty(n_time)

    positive = False
    while not positive:
        price_feature[:] = np.random.normal(0, feature_sigma, n_time)
        spot[0] = initial_spot
        for t in range(n_time - 1):
            spot[t + 1] = beta0 + beta1 * spot[t] + beta2 * price_feature[t] + np.random.normal(0, sigma)
        if np.all(spot >= 0):
            positive = True

    forward = spot + np.random.normal(0, spot / 100)
    prices = np.vstack([spot, forward]).T

    add_features = np.hstack([np.random.normal(10 * i, 2 * i, [n_time, 1]) for i in range(3, 3 + n_add_feature)])
    features = np.hstack([spot[:, None], price_feature[:, None], add_features])

    if seasonal_demand:
        demand = mean_demand + 0.5 * np.sin(np.pi * (np.arange(1, n_time + 1) + 2) / 6) * mean_demand
    else:
        demand = np.ones(n_time)

    return prices, features, demand

def generate_data():

    n_runs = 100
    processes = ['rw', 'mr']
    sigmas = [5, 20]
    n_time = 119

    index = pd.MultiIndex.from_product([processes, sigmas, range(1, n_runs + 1), range(1, n_time + 1)],
                                       names=['process', 'sigma', 'run', 'time'])
    df = pd.DataFrame(columns=['spot', 'forward', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'demand'],
                      index=index, dtype=np.float_)
    df = df.sort_index()

    for idx, _ in df.groupby(['process', 'sigma', 'run']):
        if idx[0] == 'rw':
            beta0 = 0
            beta1 = 1
        elif idx[0] == 'mr':
            beta0 = 100
            beta1 = 0.5
        else:
            raise NotImplementedError

        prices, features, demand = generate_series(n_time=n_time, beta0=beta0, beta1=beta1, sigma=idx[1],
                                                   seasonal_demand=True)
        df.loc[idx] = np.hstack([prices, features, demand[:, None]])

    return df

def simulation():

    np.random.seed(42)
    torch.manual_seed(42)

    df = generate_data()
    df.to_csv('data/simulation.csv')

    train_sizes = [24, 72]
    test_size = 48

    results = pd.DataFrame(columns=['process', 'sigma', 'run', 'train_size',
                                    'p_spot', 'p_m1', 'lasso', 'ridge', 'rnn_ieo'])
    results = results.set_index(['process', 'sigma', 'run', 'train_size'])

    for idx, run in df.groupby(['process', 'sigma', 'run']):

        prices = run[['spot', 'forward']].to_numpy()
        features = run[['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']].to_numpy()
        demand = run['demand'].to_numpy()

        # Perfect foresight costs
        c_pf = test_utils.c_pf(prices[-test_size:], demand[-test_size:])

        for train_size in train_sizes:

            size = train_size + test_size - 1
            costs = [test_utils.c_tau(prices[-test_size:], demand[-test_size:], 0),
                     test_utils.c_tau(prices[-test_size:], demand[-test_size:], 1),
                     test_regression(prices[-size:], features[-size:], demand[-size:], test_size, 'lasso'),
                     test_regression(prices[-size:], features[-size:], demand[-size:], test_size, 'ridge'),
                     test_presnet(prices[-size:], features[-size:], demand[-size:], test_size, 'rnn')]

            costs = np.array(costs)
            pe = (costs - c_pf) / c_pf * 100

            results.loc[idx + (train_size,)] = pe
            print(results.to_string())
            print(results.groupby(['process', 'sigma', 'train_size']).mean().to_string())
            results.to_pickle('results/longlongterm.pkl')

def test_presnet(prices, features, demand, test_size, cell_type):

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

    model = presnet.model.PresNet(cell_type, prices.shape[1], features.shape[1], params['n_steps'],
                                  params['n_hidden'], params['n_layers'], params['dropout'])
    train_set = presnet.dataset.PresDataset(prices[:-test_size + 1], features[:-test_size + 1],
                                            demand[:-test_size + 1], params['n_steps'])
    trainer = presnet.train.PresTrainer(model, train_set, params)
    trainer.train()

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)
    for t in range(test_size):
        idx = prices.shape[0] - test_size + t
        model.eval()
        with torch.no_grad():
            sequence = train_set.scaler.transform(features)[None, idx - params['n_steps'] + 1:idx + 1]
            logits = model(torch.tensor(sequence).float())
            signals[t, 1:] = logits.sigmoid().numpy().squeeze(axis=0) >= 0.5

    costs, _ = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    return costs.mean()

def test_prednet(prices, features, demand, test_size, cell_type):

    params = {
        'n_steps': 3,
        'n_hidden': 128,
        'n_layers': 4,
        'batch_size': 4,
        'n_epochs': 25,
        'lr': 1e-3,
        'weight_decay': 1e-6,
        'grad_clip': 10,
        'dropout': 0.0
    }

    model = prednet.model.PredNet(cell_type, features.shape[1], 1, params['n_steps'], params['n_hidden'],
                                  params['n_layers'], params['dropout'])
    train_set = prednet.dataset.PredDataset(features[:-test_size + 1, 0], features[:-test_size + 1], 1,
                                            params['n_steps'])
    trainer = prednet.train.PredTrainer(model, train_set, params)
    trainer.train()

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)
    for t in range(test_size):
        idx = prices.shape[0] - test_size + t
        model.eval()
        with torch.no_grad():
            sequence = train_set.scaler_f.transform(features)[None, idx - params['n_steps'] + 1:idx + 1]
            logits = model(torch.tensor(sequence).float())
            preds = train_set.scaler_p.inverse_transform(logits).ravel()
            signals[t, 1:] = preds >= prices[idx, 1]

    costs, _ = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    return costs.mean()

def test_dda(prices, features, demand, test_size, reg):

    model = dda.model.DDA(prices[:-test_size + 1], features[:-test_size + 1], demand[:-test_size + 1],
                          reg=reg, big_m=False)
    model.train()
    costs = model.prescribe(prices[-test_size:], model.scaler.transform(features[-test_size:]), demand[-test_size:])

    return costs.mean()

def test_regression(prices, features, demand, test_size, reg):

    params = {'alpha': np.arange(0.1, 10.1, 0.1)}
    if reg == 'lasso':
        model = Lasso(normalize=True, max_iter=1e4)
    elif reg == 'ridge':
        model = Ridge(normalize=True, max_iter=1e4)
    else:
        raise NotImplementedError
    clf = GridSearchCV(model, params, cv=2, iid=False)
    clf.fit(features[:-test_size], prices[1:-test_size + 1, 0])
    pred = clf.predict(features[-test_size:])

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)
    signals[:, 1] = pred >= prices[-test_size:, 1]
    costs, _ = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    return costs.mean()

if __name__ == '__main__':
    simulation()






