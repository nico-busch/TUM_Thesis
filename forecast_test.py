import numpy as np
import pandas as pd
import torch
import statsmodels.tsa.api as tsa
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import predictive

params = {
    'n_steps': 12,
    'n_hidden': 128,
    'n_layers': 4,
    'batch_size': 32,
    'n_epochs': 25,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'grad_clip': 10,
    'dropout': 0.0
}

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv('data/data.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)
df.index.freq = 'M'
df = df.bfill()

test_size = 36
n_fc = 4

mean = 0

for t in range(test_size - n_fc):

    # BASELINE

    naive_fc = np.repeat(df.iloc[-test_size + t - 1]['SPOT'], n_fc)

    # UNIVARIATE

    train = df.iloc[:-test_size + t]['SPOT']

    arima = tsa.arima.ARIMA(train, order=(1, 1, 1)).fit()
    arima_fc = arima.forecast(n_fc).to_numpy()

    ets = tsa.ExponentialSmoothing(train).fit()
    ets_fc = ets.forecast(n_fc).to_numpy()

    # MULTIVARIATE

    train = df.iloc[:-test_size + t][['SPOT', 'EURUSD', 'TEMP', 'M1']]

    model = tsa.VAR(train)
    var = model.fit(maxlags=12, ic='aic')
    var_fc = var.forecast(train.iloc[-var.k_ar:].to_numpy(), n_fc)[:, 0]

    # DEEP LEARNING

    features = df.iloc[:-test_size + t][['SPOT', 'EURUSD', 'M1', 'TEMP']].to_numpy()

    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)
    spot = features_std[:, 0]

    model = predictive.model.PredictiveNet(features.shape[1], n_fc,
                                           params['n_steps'], params['n_hidden'],
                                           params['n_layers'], params['dropout'])
    train_set = predictive.dataset.Dataset(spot, features_std, n_fc, params['n_steps'])
    trainer = predictive.train.Trainer(model, train_set, train_set, params)
    trainer.train()
    model.eval()
    with torch.no_grad():
        nn_fc = model(torch.tensor(features_std[None, -params['n_steps']:]).float()).numpy().squeeze() \
                * scaler.scale_[0] + scaler.mean_[0]

    targets = df.iloc[-test_size + t:-test_size + t + n_fc]['SPOT'].to_numpy()
    fc = np.vstack([naive_fc, arima_fc, ets_fc, var_fc, nn_fc]).T

    mse = ((targets[:, None] - fc) ** 2).mean(axis=0)
    mean += mse
    print(mean / (t + 1))

    results = pd.DataFrame(data=fc, columns=['naive', 'arima', 'ets', 'var', 'lstm'],
                           index=df.index[-test_size + t:-test_size + t + n_fc])