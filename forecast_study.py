import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from ax.service.managed_loop import optimize

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

import prednet

def run():

    test_size = 36
    df = pd.read_csv('data/reuters.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)

    # Univariate forecasts from R
    fc = pd.read_csv('results/forecast_uni.csv', parse_dates=['DATE'], index_col=['DATE', '.model'],
                     usecols=['DATE', '.model', '.mean'])
    fc = fc.unstack(level=1).droplevel(level=0, axis=1)
    fc.index += pd.offsets.MonthEnd()
    fc.columns = ['ARIMA', 'ETS', 'NAIVE', 'SNAIVE']

    # Multivariate forecasts from R
    fc_var = pd.read_csv('results/forecast_multi.csv', parse_dates=['DATE'], index_col=['DATE'],
                         usecols=['DATE', '.mean_SPOT'])
    fc_var.index += pd.offsets.MonthEnd()
    fc['VAR'] = fc_var

    # Actuals
    fc['ACTUAL'] = df['SPOT'].iloc[-test_size:]

    # Reorder
    fc = fc.reindex(['ACTUAL', 'NAIVE', 'ARIMA', 'ETS'], axis=1)

    # Inputs and outputs
    spot = df['SPOT'].to_numpy()
    features = df[['SPOT', 'M1', 'M2', 'M3', 'M4', 'TTF_M1', 'HENRYHUB_M1', 'BRENT_M1', 'COAL_M1', 'BCOM', 'GSCI',
                   'DXY', 'EURUSD', 'PPI', 'GAS_PROD', 'GAS_DEM', 'GAS_STO', 'SP500', 'EU10YT', 'TEMP']].to_numpy()

    # Regression models
    fc['LASSO'] = test_regression(spot, features, test_size, 'lasso')
    fc['Ridge'] = test_regression(spot, features, test_size, 'ridge')

    # Neural nets
    fc['MLP'] = test_prednet(spot, features, test_size, 'mlp')
    fc['RNN'] = test_prednet(spot, features, test_size, 'rnn')
    fc['LSTM'] = test_prednet(spot, features, test_size, 'lstm')

    # fc.to_pickle('results/forecast.pkl')
    # fc.to_csv('results/forecast.csv')


def test_regression(spot, features, test_size, reg):

    test_size = test_size + 1

    # Include spot lags in features
    spot_lag = 3
    features = np.hstack([np.vstack([features[spot_lag - lag:features.shape[0] - lag, 0]
                          for lag in range(spot_lag + 1)]).T,
                          ((features[spot_lag:, 0] - features[spot_lag - 1:-1, 0])
                           / features[spot_lag - 1:-1, 0])[:, None],
                          features[spot_lag:, spot_lag + 1:]])
    spot = spot[spot_lag:]

    # Build and train model
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)
    params = {'alpha': np.arange(0.5, 10.1, 0.1)}
    if reg == 'lasso':
        model = Lasso(max_iter=1e6)
    elif reg == 'ridge':
        model = Ridge(max_iter=1e6)
    else:
        raise NotImplementedError
    clf = GridSearchCV(model, params, cv=2, iid=False)
    clf.fit(features_std[:-test_size], spot[1:-test_size + 1])

    # Forecast
    pred = clf.predict(features_std[-test_size:])

    return pred[:-1]


def test_prednet(spot, features, test_size, cell_type, retrain=False):

    # Hyperparameters
    params = {'n_steps': 12}
    opt_params = tune_presnet(spot, features, test_size, cell_type)
    params = {**params, **opt_params}

    test_size = test_size + 1
    n_ensemble = 10
    preds = np.zeros(test_size)

    if retrain:

        for x in range(n_ensemble):

            for t in range(test_size):

                idx = spot.shape[0] - test_size + t

                # Build and train model
                model = prednet.model.PredNet(
                    cell_type, features.shape[1], 1, params['n_steps'],
                    params['n_hidden'], params['n_layers'], params['dropout'])
                train_set = prednet.dataset.PredDataset(
                    spot[:idx + 1], features[:idx + 1], 1, params['n_steps'])
                trainer = prednet.train.PredTrainer(model, train_set, params)
                trainer.train()

                # Test model
                model.eval()
                with torch.no_grad():
                    sequence = train_set.scaler_f.transform(features)[None, idx - params['n_steps'] + 1:idx + 1]
                    logits = model(torch.tensor(sequence).float())
                    preds[t] += train_set.scaler_p.inverse_transform(logits.numpy()).ravel()

    else:

        for x in range(n_ensemble):

            # Build and train model
            model = prednet.model.PredNet(
                cell_type, features.shape[1], 1, params['n_steps'],
                params['n_hidden'], params['n_layers'], params['dropout'])
            train_set = prednet.dataset.PredDataset(
                spot[:-test_size + 1], features[:-test_size + 1], 1, params['n_steps'])
            trainer = prednet.train.PredTrainer(model, train_set, params)
            trainer.train()

            # Test model
            with torch.no_grad():
                sequences = []
                for x in range(spot.shape[0] - test_size, spot.shape[0]):
                    sequences.append(train_set.scaler_f.transform(features)[x - params['n_steps'] + 1:x + 1])
                model.eval()
                logits = model(torch.tensor(sequences).float())
                preds += train_set.scaler_p.inverse_transform(logits.numpy()).ravel()

    return preds[:-1] / n_ensemble


def tune_presnet(spot, features, test_size, cell_type):

    n_steps = 12

    train_set = prednet.dataset.PredDataset(
        spot[:-test_size + 1], features[:-test_size + 1], 1, n_steps)

    def evaluate(params):

        # Time-series cross-validation
        n_splits = 3
        split = TimeSeriesSplit(n_splits, len(train_set))
        score = 0

        for train, val in split.split(list(range(len(train_set)))):

            model = prednet.model.PredNet(
                cell_type, features.shape[1], 1, n_steps,
                params['n_hidden'], params['n_layers'], params['dropout'])
            tset = Subset(train_set, train)
            vset = Subset(train_set, val)
            trainer_val = prednet.train.PredTrainer(model, tset, params, vset)
            score += trainer_val.train()

        return score / n_splits

    # Hyperparameters
    best_params, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-7, 1e-2], "log_scale": True},
            {"name": "dropout", "type": "range", "bounds": [0.0, 0.8]},
            {"name": "n_epochs", "type": "range", "bounds": [10, 100]},
            {"name": "n_hidden", "type": "range", "bounds": [32, 256]},
            {"name": "n_layers", "type": "range", "bounds": [1, 8]},
            {"name": "batch_size", "type": "range", "bounds": [4, 32]},
            {"name": "grad_clip", "type": "range", "bounds": [1e1, 1e4], "log_scale": True},
        ],
        evaluation_function=evaluate,
        objective_name='mse',
        minimize=True,
        total_trials=100,
    )

    return best_params


if __name__ == '__main__':
    run()


