import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from pa_rnn.model import RNN
from pa_rnn.dataset import Dataset
from pa_rnn.loss import cw_nll_loss
from pa_rnn.evaluate import prescribe_multiperiod


class Trainer:

    def __init__(self, prices, features, demand, test_size, params):

        self.prices = prices
        self.features = features
        self.demand = demand
        self.test_size = test_size
        self.params = params

        dataset = Dataset(prices, features, demand, test_size, params['n_steps'])
        idx = list(range(len(dataset)))
        split = test_size - prices.shape[1] + 1
        self.train_set = Subset(dataset, idx[:-split])
        self.test_set = Subset(dataset, idx[-split:])

        self.model = None

    def train(self):

        # idx = list(range(len(self.train_set)))
        # val_size = 3
        # train, val = idx[:-val_size], idx[-val_size:]

        train_loader = DataLoader(self.train_set, batch_size=self.params['batch_size'], shuffle=False)
        val_loader = DataLoader(self.test_set, batch_size=self.params['batch_size'], shuffle=False)

        best_loss = None
        best_epoch = 0
        train_losses = []
        val_losses = []

        model = RNN(self.prices.shape[1], self.features.shape[1],
                    self.params['n_steps'], self.params['n_hidden'], self.params['n_layers'], self.params['dropout'])
        optim = torch.optim.Adam(model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])

        for e in range(self.params['n_epochs']):

            train_loss = self.train_epoch(model, optim, train_loader, self.params['grad_clip'])
            val_loss = self.val(model, val_loader)

            if e == 0:
                best_loss = val_loss

            elif best_loss - val_loss >= 1e-4:
                best_loss = val_loss
                best_epoch = e

            # train_losses.append(train_loss)
            # val_losses.append(val_loss)
            val_losses.append(prescribe_multiperiod(model, self.test_set,
                                     self.prices[-self.test_size:], self.demand[-self.test_size:])[1].mean())

        # plt.plot(train_losses)
        plt.plot(val_losses)
        plt.show()

        # train_loader = DataLoader(self.train_set, batch_size=self.params['batch_size'], shuffle=False)
        #
        # model = RNN(self.prices.shape[1], self.features.shape[1],
        #             self.params['n_steps'], self.params['n_hidden'], self.params['n_layers'], self.params['dropout'])
        # optim = torch.optim.Adam(model.parameters(), lr=self.params['lr'], weight_decay=self.params['weight_decay'])
        #
        # for e in range(self.params['n_epochs']):
        #     self.train_epoch(model, optim, train_loader, self.params['grad_clip'])

        self.model = model

    @staticmethod
    def train_epoch(model, optim, train_loader, grad_clip):

        model.train()
        train_loss = 0
        n_batches = 0

        for sequences, weights in train_loader:

            logits = model(sequences)
            loss = cw_nll_loss(logits, weights)
            # criterion = torch.nn.CrossEntropyLoss()
            # loss = criterion(logits[0], weights[0].argmax(dim=1))
            train_loss += loss
            n_batches += 1

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            optim.zero_grad()

        return train_loss / n_batches

    @staticmethod
    def val(model, val_loader):

        model.eval()
        val_loss = 0
        n_batches = 0

        with torch.no_grad():
            for sequences, weights in val_loader:

                logits = model(sequences)
                loss = cw_nll_loss(logits, weights)
                val_loss += loss
                n_batches += 1

        return val_loss / n_batches

    def test(self):
        return prescribe_multiperiod(self.model, self.test_set,
                                     self.prices[-self.test_size:], self.demand[-self.test_size:])[1]






