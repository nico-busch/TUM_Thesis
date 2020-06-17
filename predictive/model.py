import torch
import numpy as np


class PredictiveNet(torch.nn.Module):

    def __init__(self, n_features, n_fc, n_steps, n_hidden, n_layers, dropout):

        super().__init__()
        self.n_features = n_features
        self.n_fc = n_fc
        self.n_steps = n_steps

        layers = []
        for l in range(n_layers):
            if l == 0:
                n_input = n_features
            else:
                n_input = n_hidden
            layers.append(torch.nn.LSTM(n_input, n_hidden, batch_first=True))
        self.lstm = torch.nn.ModuleList(layers)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(n_hidden, n_fc)

    def forward(self, sequences):

        lstm_in = sequences
        lstm_out = None
        for i, l in enumerate(self.lstm):
            residual = lstm_in
            lstm_out, _ = l(lstm_in)
            if i > 0:
                lstm_out += residual
            lstm_in = lstm_out
        dropout_out = self.dropout(lstm_out[:, -1])
        linear_out = self.linear(dropout_out)

        return linear_out

