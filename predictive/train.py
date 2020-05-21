import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Trainer:

    def __init__(self, model, train_set, val_set, params):

        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.params = params

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    def train(self):

        train_losses = []
        val_losses = []

        train_loader = DataLoader(self.train_set, batch_size=self.params['batch_size'], shuffle=False)
        val_loader = DataLoader(self.val_set, batch_size=self.params['batch_size'], shuffle=False)

        for e in range(self.params['n_epochs']):

            train_loss = self.train_epoch(train_loader)
            val_loss = self.val(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        plt.plot(train_losses)
        plt.plot(val_losses)
        plt.show(block=False)

    def train_epoch(self, train_loader):

        self.model.train()
        train_loss = 0
        n_batches = 0

        for sequences, targets in train_loader:

            logits = self.model(sequences)
            criterion = torch.nn.MSELoss()
            loss = criterion(logits, targets.view(-1, 1))
            train_loss += loss
            n_batches += targets.shape[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['grad_clip'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        return train_loss / n_batches

    def val(self, val_loader):

        self.model.eval()
        val_loss = 0
        n_batches = 0

        with torch.no_grad():
            for sequences, targets in val_loader:

                logits = self.model(sequences)
                criterion = torch.nn.MSELoss()
                loss = criterion(logits, targets.view(-1, 1))
                val_loss += loss
                n_batches += targets.shape[0]

        return val_loss / n_batches
