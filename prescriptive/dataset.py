import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, prices, features, demand, n_steps):

        self.prices = prices
        self.features = features
        self.demand = demand
        self.n_steps = n_steps

    def __len__(self):
        return self.prices.shape[0] - self.prices.shape[1] - self.n_steps + 2

    def __getitem__(self, t):

        sequences = torch.tensor(self.features[t:t + self.n_steps]).float()
        options = [np.flipud(np.flipud(self.prices[t + self.n_steps - 1:t + self.n_steps + i]).diagonal())
                   * self.demand[t + self.n_steps + i - 1]
                   for i in range(1, self.prices.shape[1])]
        weights = torch.tensor(np.array([(x.max(axis=0) - x).max() for x in options])).float()
        targets = torch.tensor(np.array([x.argmin(axis=0) == 0 for x in options])).float()

        return sequences, targets, weights


