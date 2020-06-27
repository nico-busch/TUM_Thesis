import torch


class PrescriptiveNet(torch.nn.Module):

    def __init__(self, cell_type, n_prices, n_features, n_steps, n_hidden, n_layers, dropout):

        super().__init__()
        self.cell_type = cell_type
        self.n_prices = n_prices
        self.n_features = n_features
        self.n_steps = n_steps

        if cell_type == 'mlp':
            layers = []
            n_input = n_features
            for _ in range(n_layers):
                layers.append(torch.nn.Linear(n_input, n_hidden))
                layers.append(torch.nn.ReLU())
                n_input = n_hidden
            self.cell = torch.nn.Sequential(*layers)
        elif cell_type == 'rnn':
            self.cell = torch.nn.RNN(n_features, n_hidden, num_layers=n_layers)
        elif cell_type == 'lstm':
            self.cell = torch.nn.LSTM(n_features, n_hidden, num_layers=n_layers)
        else:
            raise NotImplementedError
        self.dropout = torch.nn.Dropout(p=dropout)
        if cell_type == 'mlp':
            self.linear = torch.nn.Linear(n_hidden * n_steps, n_prices - 1)
        else:
            self.linear = torch.nn.Linear(n_hidden, n_prices - 1)

    def forward(self, x):

        if self.cell_type == 'mlp':
            cell_out = self.cell(x)
            dropout_out = self.dropout(cell_out.contiguous().view(cell_out.shape[0], -1))
        else:
            cell_out, _ = self.cell(x)
            dropout_out = self.dropout(cell_out[:, -1])

        linear_out = self.linear(dropout_out)

        return linear_out
