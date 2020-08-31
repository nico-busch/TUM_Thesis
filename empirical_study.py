import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from ax.service.managed_loop import optimize
import shap

import presnet
import dda
import test_utils
import viz


def run():

    # For reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Read prices and features from reuters
    df = pd.read_csv('data/reuters.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)
    # df = df.iloc[-72:]
    prices = df[['SPOT', 'M1', 'M2', 'M3', 'M4']].to_numpy()
    df = df.drop(['M1', 'M2', 'M3', 'M4'], axis=1)
    features = df.to_numpy()

    test_size = 36

    # Read demand from industry partner
    df_sie = pd.read_csv('gas.csv', parse_dates=['Date'], index_col='Date')
    df_sie['Day'] = df_sie.index.to_period('D').where(df_sie.index.hour >= 6,
                                                (df_sie.index - pd.DateOffset(days=1)).to_period('D'))
    df_sie['Month'] = df_sie['Day'].dt.asfreq('M')
    df_sie['Year'] = df_sie['Day'].dt.asfreq('Y')
    df_sie['Daily'] = df_sie.groupby('Day')['Demand'].transform('mean')
    df_sie['Base'] = df_sie.groupby('Month')['Daily'].transform('min')
    base = df_sie.groupby('Month')['Base'].sum().to_numpy()[-test_size:]
    peak = df_sie.groupby('Month')['Demand'].sum().to_numpy()[-test_size:] - base

    demand = np.hstack([np.tile(base[:12], 7), base])

    c_0 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 0)
    c_1 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 1)
    c_2 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 2)
    c_3 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 3)
    c_4 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 4)
    c_pf, decisions = test_utils.c_pf(prices[-test_size:], demand[-test_size:])

    # viz.forward_curve(prices[-test_size:], df.index[-test_size:])
    # viz.decision_curve(prices[-test_size:], decisions, df.index[-test_size:])
    # viz.hedge(df_sie, decisions)

    # c_dda, decisions = test_dda(prices, features, demand, test_size, reg='lasso')
    c_nn, decisions = test_presnet(prices, features, demand, test_size, 'lstm', cols=df.columns)

    viz.decision_curve(prices[-test_size:], decisions, df.index[-test_size:])
    # viz.hedge(df_sie, decisions_nn)

    # Additional spot purchase due to level difference
    c_diff = (peak * prices[-test_size:, 0]).mean()

    # Industrial partner costs
    c_sie = df_sie['Cost'].sum() / test_size - c_diff

    c_pf = c_pf + c_diff
    costs = np.array([c_0, c_1, c_2, c_3, c_4, c_nn, c_sie]) + c_diff
    pe = (costs - c_pf) / c_pf * 100

    print(pe)


def test_presnet(prices, features, demand, test_size, cell_type, n_ensemble=10, retrain=False, cols=None):

    # Hyperparameters
    params = {'n_steps': 12}
    opt_params = tune_presnet(prices, features, demand, test_size, cell_type)
    params = {**params, **opt_params}

    scores = np.zeros([test_size, prices.shape[1] - 1])

    for i in range(n_ensemble):

        # Monthly retraining
        if retrain:

            for t in range(test_size):
                idx = prices.shape[0] - test_size + t

                # Build and train model
                model = presnet.model.PresNet(
                    cell_type, prices.shape[1], features.shape[1], params['n_steps'],
                    params['n_hidden'], params['n_layers'], params['dropout'])
                train_set = presnet.dataset.PresDataset(
                    prices[:idx + 1], features[:idx + 1], demand[:idx + 1], params['n_steps'])
                trainer = presnet.train.PresTrainer(model, train_set, params)
                trainer.train()

                # Test model
                model.eval()
                with torch.no_grad():
                    sequence = train_set.scaler.transform(features)[None, idx - params['n_steps'] + 1:idx + 1]
                    logits = model(torch.tensor(sequence).float())
                    scores[t] += logits.sigmoid().numpy().squeeze(axis=0)

        else:

            # Build and train model
            model = presnet.model.PresNet(
                cell_type, prices.shape[1], features.shape[1], params['n_steps'],
                params['n_hidden'], params['n_layers'], params['dropout'])
            train_set = presnet.dataset.PresDataset(
                prices[:-test_size + 1], features[:-test_size + 1], demand[:-test_size + 1], params['n_steps'])
            trainer = presnet.train.PresTrainer(model, train_set, params)
            trainer.train()

            # Test model
            with torch.no_grad():
                sequences = []
                for x in range(features.shape[0] - test_size, features.shape[0]):
                    sequences.append(train_set.scaler.transform(features)[x - params['n_steps']:x])
                model.eval()
                logits = model(torch.tensor(sequences).float())
                scores[:] += logits.sigmoid().numpy()

            # viz.feature_importance(model, torch.tensor(sequences).float(), cols=cols)

    # Prescribe decisions
    scores = scores / n_ensemble
    signals = np.hstack([np.zeros([test_size, 1]), scores]) >= 0.5
    costs, decisions = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    viz.roc(prices[-test_size:], scores)

    return costs.mean(), decisions


def tune_presnet(prices, features, demand, test_size, cell_type):

    n_steps = 12

    train_set = presnet.dataset.PresDataset(
        prices[:-test_size + 1], features[:-test_size + 1], demand[:-test_size + 1], n_steps)

    def evaluate(params):

        print(params)

        n_splits = 3
        split = TimeSeriesSplit(n_splits, len(train_set))
        score = 0

        for train, val in split.split(list(range(len(train_set)))):

            model = presnet.model.PresNet(
                cell_type, prices.shape[1], features.shape[1], n_steps,
                params['n_hidden'], params['n_layers'], params['dropout'])
            tset = Subset(train_set, train)
            vset = Subset(train_set, val)
            trainer_val = presnet.train.PresTrainer(model, tset, params, vset)
            score += trainer_val.train()

        print(score / n_splits)

        return score / n_splits

    best_params, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-7, 1e-2], "log_scale": True},
            {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "n_epochs", "type": "range", "bounds": [10, 100]},
            {"name": "n_hidden", "type": "range", "bounds": [50, 200]},
            {"name": "n_layers", "type": "range", "bounds": [1, 8]},
            {"name": "batch_size", "type": "range", "bounds": [16, 32]},
            {"name": "grad_clip", "type": "range", "bounds": [1e1, 1e4], "log_scale": True},
        ],
        evaluation_function=evaluate,
        objective_name='accuracy',
    )

    print(best_params)
    return best_params


def test_dda(prices, features, demand, test_size, reg, retrain=False):

    # Include spot lags in features and drop futures prices
    spot_lag = prices.shape[1] - 1
    features = np.hstack([np.vstack([features[spot_lag - lag:features.shape[0] - lag, 0]
                          for lag in range(spot_lag + 1)]).T, features[spot_lag:, spot_lag + 1:]])
    prices = prices[spot_lag:]
    demand = demand[spot_lag:]

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)
    model = None

    for t in range(test_size):
        idx = prices.shape[0] - test_size + t

        # Build and train model
        if t == 0 or retrain:
            model = dda.model.DDA(prices[:idx + 1], features[:idx + 1], demand[:idx + 1], reg=reg, big_m=False)
            model.train()

        # Calculate thresholds and signals
        thresholds = np.sum(
            np.hstack([np.ones([test_size, 1]), model.scaler.transform(features[-test_size:])])[idx, :, None]
            * model.beta, axis=0)
        signals[t] = (prices[idx] <= thresholds)

    # Prescribe decisions
    costs, decisions = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)

    return costs.mean(), decisions


if __name__ == '__main__':
    run()
