import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from loss import estimated_costs, prescription_error


class Trainer:
    def __init__(self, model, spot, forward, covariate, demand, split):

        self.model = model

        prices = np.hstack([spot[:, np.newaxis], forward])
        features = np.hstack([spot[:, np.newaxis], forward, covariate])

        self.scaler = StandardScaler()
        n_split = int(round(spot.shape[0] * split))
        self.scaler.fit(features[:n_split])
        features = self.scaler.transform(features)
        self.x_train, self.y_train, self.d_train = self.generate_seq(
            prices[:n_split], features[:n_split], demand[:n_split], self.model.n_steps)
        self.x_test, self.y_test, self.d_test = self.generate_seq(
            prices[n_split:], features[n_split:], demand[n_split:], self.model.n_steps)

    @staticmethod
    def generate_seq(prices, features, demand, n_steps):

        n_time = prices.shape[0]
        n_forwards = prices.shape[1] - 1
        x, y, d = [], [], []

        for p in range(n_forwards):
            y_ = []
            for t in range(n_time - n_steps - n_forwards + 1):
                y_.append(np.diag(np.flipud(prices[t + n_steps - 1:t + n_steps + p + 1])))
            y.append(torch.tensor(y_).float())

        for t in range(n_time - n_steps - n_forwards + 1):
            x.append(features[t:t + n_steps])
            d.append(demand[t + n_steps:t + n_steps + n_forwards])

        x = torch.tensor(x).float()
        d = torch.tensor(d).float()

        return x, y, d

    def train(self, n_epochs=10):

        optim = torch.optim.Adam(self.model.parameters())

        for t in range(n_epochs):
            optim.zero_grad()
            weights = self.model.forward(self.x_train)
            loss = estimated_costs(weights, self.y_train, self.d_train)
            loss.backward()
            optim.step()

    def test(self):
        weights = self.model.forward(self.x_test)
        return prescription_error(weights, self.y_test, self.d_test)
