import numpy as np
import torch
from torch.utils.data import Dataset


class PRNNDataset(Dataset):

    def __init__(self, prices, covariates, demand, n_steps):

        self.n_time = prices.shape[0]
        self.n_prices = prices.shape[1]
        self.n_steps = n_steps

        self.prices = prices
        self.covariates = covariates
        self.demand = demand

    def __len__(self):
        return self.n_time - self.n_steps - self.n_prices + 2

    def __getitem__(self, t):
        features = np.hstack([self.prices, self.covariates])
        x = torch.tensor(features[t:t + self.n_steps]).float()
        y, d = [], []
        for p in range(1, self.n_prices):
            y.append(torch.tensor(np.diag(np.flipud(self.prices[t + self.n_steps - 1:t + self.n_steps + p]))).float())
            d.append(torch.tensor(self.demand[t + self.n_steps - 1 + p]).float())
        return x, y, d


