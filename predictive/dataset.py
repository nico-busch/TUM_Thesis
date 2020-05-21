import torch
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(self, features, n_steps):

        self.features = features
        self.n_steps = n_steps


    def __len__(self):
        return self.features.shape[0] - self.n_steps

    def __getitem__(self, t):

        sequences = torch.tensor(self.features[t:t + self.n_steps]).float()
        targets = torch.tensor(self.features[t + self.n_steps, 0]).float()

        return sequences, targets
