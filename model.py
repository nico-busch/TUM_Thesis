import torch


class NN(torch.nn.Module):

    def __init__(self, n_steps, n_input, n_forwards, n_hidden):

        super().__init__()
        self.n_steps = n_steps
        self.n_forwards = n_forwards
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.lstm1 = torch.nn.LSTM(n_input, n_hidden, batch_first=True)
        self.lstm2 = torch.nn.LSTM(n_hidden, n_hidden, batch_first=True)
        self.linear = [torch.nn.Linear(n_hidden * n_steps, f + 2) for f in range(n_forwards)]
        self.softmax = [torch.nn.Softmax(dim=1) for f in range(n_forwards)]

    def forward(self, x):

        lstm1_out, _ = self.lstm1(x)
        lstm2_out, _ = self.lstm2(lstm1_out)

        out = []
        for f in range(self.n_forwards):
            linear_out = self.linear[f](lstm2_out.contiguous().view(lstm2_out.shape[0], -1))
            softmax_out = self.softmax[f](linear_out)
            out.append(softmax_out)

        return out
