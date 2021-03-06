import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import PowerTransformer


class PresDataset(Dataset):

    def __init__(self, prices, features, demand, n_steps, scaler=None):

        self.prices = prices
        self.features = features
        self.demand = demand
        self.n_steps = n_steps

        # Transform inputs
        if scaler is None:
            # Apply box-cox instead of yeo-johnson for strictly positive data since it's numerically more stable
            if (features < 0).any():
                self.scaler = PowerTransformer(standardize=True, method='yeo-johnson')
            else:
                self.scaler = PowerTransformer(standardize=True, method='box-cox')
            self.features_std = self.scaler.fit_transform(features)
        else:
            self.scaler = scaler
            self.features_std = self.scaler.transform(features)

    def __len__(self):
        return self.prices.shape[0] - self.prices.shape[1] - self.n_steps + 2

    def __getitem__(self, t):

        # Apply sliding window method for input windows
        sequences = torch.tensor(self.features_std[t:t + self.n_steps]).float()

        # Calculate weights for cost-sensitive classification
        options = [np.flipud(np.flipud(self.prices[t + self.n_steps - 1:t + self.n_steps + i]).diagonal())
                   * self.demand[t + self.n_steps + i - 1]
                   for i in range(1, self.prices.shape[1])]
        weights = torch.tensor([abs((x[0] - x[1:].min())) for x in options]).float()

        # Calculate targets
        targets = torch.tensor([x.argmin(axis=0) == 0 for x in options]).float()

        return sequences, targets, weights
