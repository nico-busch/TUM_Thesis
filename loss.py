import torch


def expected_costs(weights, y, d):
    ec = []
    for p in range(len(weights)):
        ec.append(torch.sum(weights[p] * y[p] * d[p].view(-1, 1), dim=1))
    return torch.mean(torch.sum(torch.stack(ec, dim=1), dim=1))


def prescription_error(decisions, y, d):
    pe = []
    for p in range(len(decisions)):
        c = torch.squeeze(torch.gather(y[p], 1, decisions[p])) * d[p].view(-1, 1)
        c_pf = torch.min(y[p], dim=1)[0] * d[p].view(-1, 1)
        pe.append((c - c_pf) / c_pf)
    return torch.mean(torch.stack(pe, dim=1))
