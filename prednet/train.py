import torch
from torch.utils.data import DataLoader

from loss import l1_regularizer


class PredTrainer:

    def __init__(self, model, train_set, params, val_set=None):

        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.params = params

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    def train(self):

        train_loader = DataLoader(self.train_set, batch_size=self.params['batch_size'], shuffle=False)
        for e in range(self.params['n_epochs']):
            self.train_epoch(train_loader)

        if self.val_set is not None:
            val_loader = DataLoader(self.val_set, batch_size=len(self.val_set), shuffle=False)
            return self.val(val_loader)

    def train_epoch(self, train_loader):

        self.model.train()
        train_loss = 0
        n_samples = 0

        for sequences, targets in train_loader:

            logits = self.model(sequences)
            criterion = torch.nn.MSELoss()
            loss = criterion(logits, targets)
            # loss += l1_regularizer(self.model, 1e-5)
            train_loss += loss * targets.numel()
            n_samples += targets.numel()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params['grad_clip'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        return train_loss / n_samples

    def val(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            for sequences, targets in val_loader:

                logits = self.model(sequences)
                criterion = torch.nn.MSELoss()
                loss = criterion(logits, targets)

        return loss.item()
