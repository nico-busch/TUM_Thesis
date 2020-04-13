import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from loss import expected_costs, prescription_error


class PRNNTrainer:

    def __init__(self, model, train_set, val_set, n_epochs=100, batch_size=12):

        self.model = model
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def train(self):

        optim = torch.optim.Adam(self.model.parameters(), weight_decay=0)
        train_loss = []
        val_loss = []

        for e in range(self.n_epochs):
            n_batches = 0
            train_pe = 0

            for x, y, d in self.train_loader:
                optim.zero_grad()
                weights = self.model.forward(x)
                loss = expected_costs(weights, y, d)
                loss.backward()
                optim.step()

                decisions = []
                for p in range(len(weights)):
                    decisions.append(torch.argmax(weights[p], dim=1).view(-1, 1))
                train_pe += prescription_error(decisions, y, d)
                n_batches += 1

            train_pe = train_pe / n_batches
            train_loss.append(train_pe)

            n_batches = 0
            val_pe = 0

            for x, y, d in self.val_loader:
                weights = self.model.forward(x)
                decisions = []
                for p in range(len(weights)):
                    decisions.append(torch.argmax(weights[p], dim=1).view(-1, 1))
                val_pe += prescription_error(decisions, y, d)
                n_batches += 1

            val_pe = val_pe / n_batches
            val_loss.append(val_pe)

        plt.plot(train_loss, label='train')
        plt.plot(val_loss, label='val')
        plt.legend()
        plt.show()



