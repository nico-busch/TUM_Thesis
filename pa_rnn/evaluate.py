import numpy as np
import torch

def prescribe_multiperiod(model, dataset, prices, demand):

    costs = np.zeros(demand.size)
    satisfied = np.zeros(demand.size, dtype=np.bool_)

    for t in range(demand.size):

        model.eval()
        with torch.no_grad():

            if not satisfied[t]:
                costs[t] += prices[t, 0] * demand[t]
                satisfied[t] = True

            if t <= demand.size - model.n_prices:

                sequence, _ = dataset[t]
                probs = model(sequence[None, :])
                signals = np.hstack([probs[p].softmax(dim=1).numpy().squeeze().argmax() == 0
                                     for p in range(len(probs))])

                for tau in range(1, model.n_prices):

                    if not satisfied[t + tau] and signals[tau - 1]:
                        costs[t] += prices[t, tau] * demand[t + tau]
                        satisfied[t + tau] = True

    return signals, costs
