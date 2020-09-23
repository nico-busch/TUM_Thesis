import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import PowerTransformer


class PredDataset(Dataset):

    def __init__(self, spot, features, n_fc, n_steps, scaler_f=None, scaler_p=None):

        self.spot = spot
        self.features = features
        self.n_fc = n_fc
        self.n_steps = n_steps

        # Transform inputs and outputs
        # Apply box-cox instead of yeo-johnson for strictly positive data since it's numerically more stable
        if scaler_f is None or scaler_p is None:
            if (features < 0).any():
                self.scaler_f = PowerTransformer(standardize=True, method='yeo-johnson')
                self.scaler_p = PowerTransformer(standardize=True, method='yeo-johnson')
            else:
                self.scaler_f = PowerTransformer(standardize=True, method='box-cox')
                self.scaler_p = PowerTransformer(standardize=True, method='box-cox')
            self.features_std = self.scaler_f.fit_transform(features)
            self.spot_std = self.scaler_p.fit_transform(spot[:, None]).ravel()
        else:
            self.scaler_f = scaler_f
            self.scaler_p = scaler_p
            self.features_std = self.scaler_f.transform(features)
            self.spot_std = self.scaler_p.transform(spot[:, None]).ravel()

    def __len__(self):
        return self.features.shape[0] - self.n_steps - self.n_fc + 1

    def __getitem__(self, t):

        # Apply sliding window method for input and output windows
        sequences = torch.tensor(self.features_std[t:t + self.n_steps]).float()
        targets = torch.tensor(self.spot_std[t + self.n_steps:t + self.n_steps + self.n_fc]).float()

        return sequences, targets
