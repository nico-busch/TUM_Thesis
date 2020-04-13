import torch


class PRNN(torch.nn.Module):

    def __init__(self, n_steps, n_features, n_prices, n_hidden):

        super().__init__()
        self.n_steps = n_steps
        self.n_prices = n_prices
        self.n_input = n_features
        self.n_hidden = n_hidden
        self.lstm1 = torch.nn.LSTM(n_features, n_hidden, batch_first=True)
        self.lstm2 = torch.nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.linear = [torch.nn.Linear(n_hidden * n_steps, p + 2) for p in range(n_prices - 1)]
        self.softmax = [torch.nn.Softmax(dim=1) for f in range(n_prices - 1)]

    def forward(self, x):

        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)

        out = []
        for p in range(self.n_prices - 1):
            linear_out = self.linear[p](lstm2_out.contiguous().view(lstm2_out.shape[0], -1))
            softmax_out = self.softmax[p](linear_out)
            out.append(softmax_out)

        return out
