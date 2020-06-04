import torch
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, spot, features, n_fc, n_steps):

        self.spot = spot
        self.features = features
        self.n_fc = n_fc
        self.n_steps = n_steps

    def __len__(self):
        return self.features.shape[0] - self.n_steps - self.n_fc + 1

    def __getitem__(self, t):

        sequences = torch.tensor(self.features[t:t + self.n_steps]).float()
        targets = torch.tensor(self.spot[t + self.n_steps:t + self.n_steps + self.n_fc]).float()

        return sequences, targets
