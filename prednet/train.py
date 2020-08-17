import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


class PredTrainer:

    def __init__(self, model, train_set, params):

        self.model = model
        self.train_set = train_set
        self.params = params

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    def train(self):

        train_loader = DataLoader(self.train_set, batch_size=self.params['batch_size'], shuffle=False)
        for e in range(self.params['n_epochs']):
            self.train_epoch(train_loader)

    def train_epoch(self, train_loader):

        self.model.train()
        train_loss = 0
        n_batches = 0

        for sequences, targets in train_loader:

            logits = self.model(sequences)
            criterion = torch.nn.MSELoss()
            loss = criterion(logits, targets)
            train_loss += loss
            n_batches += 1
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
                loss = criterion(logits, targets)
                val_loss += loss
                n_batches += 1

        return val_loss / n_batches
