import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
from ax.service.managed_loop import optimize

import presnet
import dda
import test_utils
import viz


def run():

    # Read prices and features from reuters
    df = pd.read_csv('data/reuters.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)
    prices = df[['SPOT', 'M1', 'M2', 'M3', 'M4']].to_numpy()
    df = df[['SPOT', 'M1', 'M2', 'M3', 'M4', 'TTF_M1', 'HENRYHUB_M1', 'BRENT_M1', 'COAL_M1', 'BCOM', 'GSCI', 'DXY',
             'EURUSD', 'PPI', 'GAS_PROD', 'GAS_DEM', 'GAS_STO', 'SP500', 'EU10YT', 'TEMP']]
    features = df.to_numpy()

    test_size = 36

    # Read demand from industry partner
    df_sie = pd.read_csv('data/gas.csv', parse_dates=['Date'], index_col='Date')
    df_sie[['Demand', 'Unhedged', 'Hedged', 'Cost']] = df_sie[['Demand', 'Unhedged', 'Hedged', 'Cost']] / 10 ** 6
    df_sie['Day'] = df_sie.index.to_period('D').where(df_sie.index.hour >= 6,
                                                (df_sie.index - pd.DateOffset(days=1)).to_period('D'))
    df_sie['Month'] = df_sie['Day'].dt.asfreq('M')
    df_sie['Year'] = df_sie['Day'].dt.asfreq('Y')
    df_sie['Daily'] = df_sie.groupby('Day')['Demand'].transform('mean')
    df_sie['Base'] = df_sie.groupby('Month')['Daily'].transform('min')
    base = df_sie.groupby('Month')['Base'].sum().to_numpy()[-test_size:]
    peak = df_sie.groupby('Month')['Demand'].sum().to_numpy()[-test_size:] - base
    demand = np.hstack([np.tile(base[:12], 7), base])

    # Additional spot purchase due to level difference
    c_diff = (peak * prices[-test_size:, 0])

    # Industrial partner costs
    c_sie = df_sie.groupby('Month')['Cost'].sum() - c_diff

    results = pd.DataFrame(columns=[
        'p_ub', 'p_pf', 'p_spot', 'p_m1', 'p_m2', 'p_m3', 'p_m4',
        'siemens',
        'dda_ml1', 'dda_ml2',
        'mlp', 'rnn', 'lstm'
    ])

    # Baseline
    c_0 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 0)
    c_1 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 1)
    c_2 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 2)
    c_3 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 3)
    c_4 = test_utils.c_tau(prices[-test_size:], demand[-test_size:], 4)
    c_pf, decisions = test_utils.c_pf(prices[-test_size:], demand[-test_size:])
    c_ub, decisions = test_utils.c_ub(prices[-test_size:], demand[-test_size:])

    # viz.forward_curve(prices[-test_size:], df.index[-test_size:])

    # Regression
    c_dda1, pa_dda1 = test_dda(prices, features, demand, test_size, reg='lasso', df=df)
    c_dda2, pa_dda2 = test_dda(prices, features, demand, test_size, reg='ridge', df=df)

    # Neural Nets
    c_nn1, pa_nn1 = test_presnet(prices, features, demand, test_size, 'mlp', df=df)
    c_nn2, pa_nn2 = test_presnet(prices, features, demand, test_size, 'rnn', df=df)
    c_nn3, pa_nn3 = test_presnet(prices, features, demand, test_size, 'lstm', df=df)

    # Build result table
    costs = np.vstack([c_ub, c_pf, c_0, c_1, c_2, c_3, c_4, c_sie,
                       c_dda1, c_dda2, c_nn1, c_nn2, c_nn3]).T + c_diff[:, None]
    pa = np.hstack([np.nan] * 8 + [pa_dda1, pa_dda2, pa_nn1, pa_nn2, pa_nn3])
    c_pf = c_pf + c_diff
    results.loc['2017'] = costs[:12].sum(axis=0)
    results.loc['2018'] = costs[12:24].sum(axis=0)
    results.loc['2019'] = costs[24:36].sum(axis=0)
    results.loc['total'] = costs.sum(axis=0)
    results.loc['avg'] = costs.sum(axis=0) / df_sie['Demand'].sum()
    results.loc['pe'] = (costs.sum(axis=0) - c_pf.sum()) / c_pf.sum() * 100
    results.loc['pa'] = pa * 100
    results.loc['savings'] = (c_sie + c_diff).sum() - costs.sum(axis=0)

    # results.to_pickle('results/empirical_study.pkl')
    # results.to_csv('results/empirical_study.csv')

def test_presnet(prices, features, demand, test_size, cell_type, n_ensemble=1, df=None, retrain=False):

    # Hyperparameters
    params = {'n_steps': 12}
    # opt_params = tune_presnet(prices, features, demand, test_size, cell_type)
    opt_params = {'lr': 5e-4,
                  'weight_decay': 1e-05,
                  'dropout': 0.5,
                  'n_epochs': 100,
                  'n_hidden': 64,
                  'n_layers': 4,
                  'batch_size': 16,
                  'grad_clip': 10}
    params = {**params, **opt_params}

    scores = np.zeros([test_size, prices.shape[1] - 1])
    model = None

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
            test_set = presnet.dataset.PresDataset(
                prices[-test_size - params['n_steps'] + 1:], features[-test_size - params['n_steps'] + 1:],
                demand[-test_size - params['n_steps'] + 1:], params['n_steps'], scaler=train_set.scaler)
            trainer = presnet.train.PresTrainer(model, train_set, params, val_set=test_set)
            trainer.train()

            # Test model
            with torch.no_grad():
                sequences = []
                for x in range(prices.shape[0] - test_size, prices.shape[0]):
                    sequences.append(train_set.scaler.transform(features)[x - params['n_steps'] + 1:x + 1])
                model.eval()
                logits = model(torch.tensor(sequences).float())
                scores[:] += logits.sigmoid().numpy()

    # Prescribe decisions
    scores = scores / n_ensemble
    signals = np.hstack([np.zeros([test_size, 1]), scores]) >= 0.5
    costs, decisions = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)
    pa = test_utils.prescription_accuracy(prices[-test_size:], demand[-test_size:], signals).get('pa_total')

    # Visualizations
    viz.roc(prices[-test_size:], scores)
    viz.pr(prices[-test_size:], scores)
    viz.decision_curve(prices[-test_size:], decisions, df.index[-test_size:])
    viz.feature_importance(model, features, test_size, cols=df.columns)

    return costs, pa


def tune_presnet(prices, features, demand, test_size, cell_type):

    n_steps = 12
    train_set = presnet.dataset.PresDataset(
        prices[:-test_size + 1], features[:-test_size + 1], demand[:-test_size + 1], n_steps)

    def evaluate(params):

        # Time-series cross-validation
        n_splits = 2
        split = TimeSeriesSplit(n_splits, len(train_set))
        score = 0

        for train, val in split.split(list(range(len(train_set)))):

            model = presnet.model.PresNet(
                cell_type, prices.shape[1], features.shape[1], n_steps,
                params['n_hidden'], params['n_layers'], params['dropout'])
            tset = Subset(train_set, train)
            vset = Subset(train_set, val)
            trainer_val = presnet.train.PresTrainer(model, tset, params, vset)
            score += trainer_val.train().get('wbce')

        return score / n_splits

    # Bayesian optimization
    best_params, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-2], "log_scale": True},
            {"name": "weight_decay", "type": "range", "bounds": [1e-7, 1e-2], "log_scale": True},
            {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
            {"name": "n_epochs", "type": "range", "bounds": [10, 100]},
            {"name": "n_hidden", "type": "range", "bounds": [50, 200]},
            {"name": "n_layers", "type": "range", "bounds": [1, 8]},
            {"name": "batch_size", "type": "range", "bounds": [4, 32]},
            {"name": "grad_clip", "type": "range", "bounds": [1e1, 1e4], "log_scale": True},
        ],
        evaluation_function=evaluate,
        objective_name='wbce',
        minimize=True,
        total_trials=100,
    )

    return best_params


def test_dda(prices, features, demand, test_size, reg, df=None, retrain=True):

    # Include spot lags and spot returns in features and drop futures prices
    spot_lag = prices.shape[1] - 1
    features = np.hstack([np.vstack([features[spot_lag - lag:features.shape[0] - lag, 0]
                          for lag in range(spot_lag + 1)]).T,
                          ((features[spot_lag:, 0] - features[spot_lag - 1:-1, 0])
                           / features[spot_lag - 1:-1, 0])[:, None],
                          features[spot_lag:, spot_lag + 1:]])
    prices = prices[spot_lag:]
    demand = demand[spot_lag:]

    signals = np.zeros([test_size, prices.shape[1]], dtype=np.bool_)
    model = None

    for t in range(test_size):
        idx = prices.shape[0] - test_size + t
        # lag = 12
        lag = idx

        # Build and train model
        if t == 0 or retrain:
            model = dda.model.DDA(
                prices[idx - lag:idx + 1], features[idx - lag:idx + 1], demand[idx - lag:idx + 1], reg=reg, big_m=False)
            model.train()

        # Calculate thresholds and signals
        thresholds = np.sum(
            np.hstack([np.ones([test_size, 1]), model.scaler.transform(features[-test_size:])])[t, :, None]
            * model.beta, axis=0)
        signals[t] = (prices[idx] <= thresholds)

    # Prescribe decisions
    costs, decisions = test_utils.c_prescribe(prices[-test_size:], demand[-test_size:], signals)
    pa = test_utils.prescription_accuracy(prices[-test_size:], demand[-test_size:], signals).get('pa_total')

    # Visualizations
    viz.decision_curve(prices[-test_size:], decisions, df.index[-test_size:])

    return costs, pa


if __name__ == '__main__':
    run()
