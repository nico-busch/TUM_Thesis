import torch


class RNN(torch.nn.Module):

    def __init__(self, n_prices, n_features, n_steps, n_hidden, n_layers, dropout):

        super().__init__()
        self.n_prices = n_prices
        self.n_features = n_features
        self.n_steps = n_steps

        layers = []
        for l in range(n_layers):
            if l == 0:
                n_input = n_features
            else:
                n_input = n_hidden
            layers.append(torch.nn.LSTM(n_input, n_hidden, batch_first=True))
        self.lstm = torch.nn.Sequential(*layers)

        self.dropout = torch.nn.Dropout(p=dropout)

        self.linear = [torch.nn.Linear(n_hidden, p + 2) for p in range(n_prices - 1)]

    def forward(self, features):

        lstm_in = features
        lstm_out = None
        for l in range(len(self.lstm)):
            residual = lstm_in
            lstm_out, _ = self.lstm[l](lstm_in)
            if l > 0:
                lstm_out += residual
            lstm_in = lstm_out

        dropout_out = self.dropout(lstm_out[:, -1])

        out = []
        for p in range(self.n_prices - 1):
            out.append(self.linear[p](dropout_out.contiguous().view(dropout_out.shape[0], -1)))

        return out
