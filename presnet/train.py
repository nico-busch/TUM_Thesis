import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from presnet.loss import weighted_bce_loss, l1_regularizer


class PresTrainer:

    def __init__(self, model, train_set, params, val_set=None, weighted=True):

        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.params = params
        self.weighted = weighted

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    def train(self):

        train_loader = DataLoader(self.train_set, batch_size=self.params['batch_size'], shuffle=False)
        for e in range(self.params['n_epochs']):
            self.train_epoch(train_loader)

        if self.val_set is not None:
            val_loader = DataLoader(self.val_set, batch_size=self.params['batch_size'], shuffle=False)
            return self.val(val_loader)

    def train_epoch(self, train_loader):

        self.model.train()
        train_loss = 0
        n_batches = 0

        for sequences, targets, weights in train_loader:

            logits = self.model(sequences)
            weights = weights if self.weighted else 1
            loss = weighted_bce_loss(logits, targets, weights)
            # loss += l1_regularizer(self.model)
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
        n_samples = 0

        with torch.no_grad():
            for sequences, targets, weights in val_loader:

                logits = self.model(sequences)
                preds = logits.sigmoid() >= 0.5
                loss = preds.eq(targets >= 0.5).sum().item()
                val_loss += loss
                n_samples += targets.numel()

        return val_loss / n_samples







