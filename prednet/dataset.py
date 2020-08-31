import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import PowerTransformer


class PredDataset(Dataset):

    def __init__(self, spot, features, n_fc, n_steps):

        self.spot = spot
        self.features = features
        self.n_fc = n_fc
        self.n_steps = n_steps

        # Transform inputs and outputs
        self.scaler_f = PowerTransformer(method='yeo-johnson')
        self.features_std = self.scaler_f.fit_transform(features)
        self.scaler_p = PowerTransformer(method='yeo-johnson')
        self.spot_std = self.scaler_p.fit_transform(spot[:, None]).ravel()

    def __len__(self):
        return self.features.shape[0] - self.n_steps - self.n_fc + 1

    def __getitem__(self, t):

        # Apply sliding window method for input and output windows
        sequences = torch.tensor(self.features_std[t:t + self.n_steps]).float()
        targets = torch.tensor(self.spot_std[t + self.n_steps:t + self.n_steps + self.n_fc]).float()

        return sequences, targets
