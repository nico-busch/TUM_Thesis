import torch


def estimated_costs(weights, y, d):
    c = []
    for p in range(len(y)):
        c.append(torch.sum(weights[p] * y[p] * d.view(-1, 1)[p], dim=1))
    return torch.mean(torch.sum(torch.stack(c, dim=1), dim=1))


def prescription_error(weights, y, d):
    pe = []
    for p in range(len(y)):
        c = torch.squeeze(torch.gather(y[p], 1, torch.argmax(weights[p], dim=1).view(-1, 1)))
        c_pf, _ = torch.min(y[p], dim=1)
        pe.append((c - c_pf) / c_pf)
    return torch.mean(torch.stack(pe, dim=1))
