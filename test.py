import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler

import predictive

np.random.seed(42)
torch.manual_seed(42)

params = {
    'n_steps': 12,
    'n_hidden': 128,
    'n_layers': 4,
    'batch_size': 32,
    'n_epochs': 100,
    'lr': 1e-3,
    'weight_decay': 1e-6,
    'grad_clip': 10,
    'dropout': 0.0
}

df = pd.read_csv('data/data.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)
df = df.bfill()
features = df[['SPOT']].to_numpy()

test_size = 12

scaler = StandardScaler()
scaler.fit(features[:-test_size])
features_std = scaler.transform(features)

model = predictive.model.PredictiveNet(features.shape[1], params['n_steps'], params['n_hidden'], params['n_layers'],
                                       params['dropout'])
dataset = predictive.dataset.Dataset(features_std, params['n_steps'])
idx = list(range(len(dataset)))
train_set = Subset(dataset, idx[:-test_size])
val_set = Subset(dataset, idx[-test_size:])
trainer = predictive.train.Trainer(model, train_set, val_set, params)
trainer.train()


