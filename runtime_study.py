import numpy as np
import torch
import pandas as pd
import timeit

import dda
import presnet
import numerical_study

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)
runs = 100
train_sizes = np.arange(10, 210, 10)

results = pd.DataFrame(columns=['run', 'train_size', 'dda_bd', 'mlp_ieo', 'rnn_ieo', 'lstm_ieo'])
results = results.set_index(['run', 'train_size'])

# Test runtimes
for i in range(1, runs + 1):

    # Generate synthetic dataset
    prices, features, demand = numerical_study.generate_series(200)

    for train_size in train_sizes:

        # DDA
        start = timeit.default_timer()
        model = dda.model.DDA(prices[:train_size], features[:train_size], demand[:train_size], reg=None, big_m=False)
        model.train()
        results.loc[(i, train_size), 'dda_bd'] = timeit.default_timer() - start

        # Hyperparameters
        params = {
            'n_steps': 1,
            'n_hidden': 128,
            'n_layers': 2,
            'batch_size': 32,
            'n_epochs': 50,
            'lr': 1e-3,
            'weight_decay': 1e-6,
            'grad_clip': 1e3,
            'dropout': 0.1
        }

        # ANN dataset
        train_set = presnet.dataset.PresDataset(
            prices[:train_size], features[:train_size], demand[:train_size], params['n_steps'])

        # MLP
        start = timeit.default_timer()
        model = presnet.model.PresNet('mlp', prices.shape[1], features.shape[1], params['n_steps'],
                                      params['n_hidden'], params['n_layers'], params['dropout'])
        trainer = presnet.train.PresTrainer(model, train_set, params, weighted=True)
        trainer.train()
        results.loc[(i, train_size), 'mlp_ieo'] = timeit.default_timer() - start

        # RNN
        start = timeit.default_timer()
        model = presnet.model.PresNet('rnn', prices.shape[1], features.shape[1], params['n_steps'],
                                      params['n_hidden'], params['n_layers'], params['dropout'])
        trainer = presnet.train.PresTrainer(model, train_set, params, weighted=True)
        trainer.train()
        results.loc[(i, train_size), 'rnn_ieo'] = timeit.default_timer() - start

        # LSTM
        start = timeit.default_timer()
        model = presnet.model.PresNet('lstm', prices.shape[1], features.shape[1], params['n_steps'],
                                      params['n_hidden'], params['n_layers'], params['dropout'])
        trainer = presnet.train.PresTrainer(model, train_set, params, weighted=True)
        trainer.train()
        results.loc[(i, train_size), 'lstm_ieo'] = timeit.default_timer() - start

    # results.to_pickle('results/runtimes.pkl')
    # results.to_csv('results/runtimes.csv')
