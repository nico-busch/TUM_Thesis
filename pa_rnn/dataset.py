import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, prices, features, demand, test_size, n_steps):

        self.prices = prices
        self.features = features
        self.demand = demand
        self.test_size = test_size
        self.n_steps = n_steps

        self.features_std = (features - features[:-test_size].mean(axis=0)) / features[:-test_size].std(axis=0)
        self.weights = [np.asarray([np.fliplr(prices[i:, :j + 1]).diagonal()
                                    for i in range(prices.shape[0] - prices.shape[1] + 1)])
                        * demand[j:demand.shape[0] - prices.shape[1] + j + 1, None]
                        for j in range(1, prices.shape[1])]

    def __len__(self):
        return self.prices.shape[0] - self.prices.shape[1] - self.n_steps + 2

    def __getitem__(self, t):

        sequences = torch.tensor(self.features_std[t:t + self.n_steps]).float()
        weights = [torch.tensor(x[t + self.n_steps - 1]) for x in self.weights]

        return sequences, weights


