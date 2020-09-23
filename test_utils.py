import numpy as np
import gurobipy as gp
from gurobipy import GRB


# Cost under perfect foresight
def c_pf(prices, demand):

    T = prices.shape[0]
    F = prices.shape[1] - 1

    p = {(t, tau): prices[t - 1, tau] for t in range(1, T + 1) for tau in range(F + 1)}
    d = {t: demand[t - 1] for t in range(1, T + 1)}

    m = gp.Model()
    m.Params.outputFlag = 0
    q = m.addVars(range(1, T + 1), range(F + 1))

    m.setObjective(gp.quicksum(p[t, tau] * q[t, tau]
                               for t in range(1, T + 1) for tau in range(F + 1) if tau <= T - t) / T, GRB.MINIMIZE)

    m.addConstrs(gp.quicksum(q[t - tau, tau] for tau in range(F + 1) if tau <= t - 1) == d[t]
                 for t in range(1, T + 1))

    m.optimize()

    costs = np.zeros(T)
    for t in range(1, T + 1):
        for tau in range(F + 1):
            if tau <= T - t:
                costs[t + tau - 1] += p[t, tau] * q[t, tau].x
    decisions = np.array([1 if val.x >= 1e-3 else 0 for val in q.values()]).reshape([T, F + 1])

    return costs, decisions


# Upper bound cost
def c_ub(prices, demand):

    T = prices.shape[0]
    F = prices.shape[1] - 1

    p = {(t, tau): prices[t - 1, tau] for t in range(1, T + 1) for tau in range(F + 1)}
    d = {t: demand[t - 1] for t in range(1, T + 1)}

    m = gp.Model()
    m.Params.outputFlag = 0
    q = m.addVars(range(1, T + 1), range(F + 1))

    m.setObjective(gp.quicksum(p[t, tau] * q[t, tau]
                               for t in range(1, T + 1) for tau in range(F + 1) if tau <= T - t) / T, GRB.MAXIMIZE)

    m.addConstrs(gp.quicksum(q[t - tau, tau] for tau in range(F + 1) if tau <= t - 1) == d[t]
                 for t in range(1, T + 1))

    m.optimize()

    costs = np.zeros(T)
    for t in range(1, T + 1):
        for tau in range(F + 1):
            if tau <= T - t:
                costs[t + tau - 1] += p[t, tau] * q[t, tau].x
    decisions = np.array([1 if val.x >= 1e-3 else 0 for val in q.values()]).reshape([T, F + 1])

    return costs, decisions


# Tau-only policy
def c_tau(prices, demand, tau):

    if tau == 0:
        costs = prices[:, 0] * demand

    else:
        costs = np.zeros(prices.shape[0])
        costs[tau:] += prices[:-tau, tau] * demand[tau:]
        costs[:tau] += prices[:tau, 0] * demand[:tau]

    return costs


# Util to prescribe optimal decisions from raw signals
def c_prescribe(prices, demand, signals):

    costs = np.zeros(prices.shape[0])
    satisfied = np.zeros(prices.shape[0], dtype=np.bool_)
    decisions = np.zeros(prices.shape, dtype=np.bool_)

    for t in range(prices.shape[0]):

        if not satisfied[t]:
            costs[t] += prices[t, 0] * demand[t]
            satisfied[t] = True
            decisions[t, 0] = True

        for tau in range(1, prices.shape[1]):
            if tau >= prices.shape[0] - t:
                break
            if not satisfied[t + tau] and signals[t, tau]:
                costs[t + tau] += prices[t, tau] * demand[t + tau]
                satisfied[t + tau] = True
                decisions[t, tau] = True

    return costs, decisions


# Calculate total PA and per-class PA
def prescription_accuracy(prices, demand, signals):

    targets = []
    weights = []
    for t in range(prices.shape[0] - prices.shape[1] + 1):
        options = [np.flipud(np.flipud(prices[t:t + i + 1]).diagonal())
                   * demand[t + i] for i in range(1, prices.shape[1])]
        targets.append(np.array([int(x.argmin(axis=0) == 0) for x in options]))
        weights.append(np.array([abs((x[0] - x[1:].min())) for x in options]))
    targets = np.vstack(targets)
    weights = np.vstack(weights)
    signals = signals[:-prices.shape[1] + 1, 1:]

    pa_total = weights[signals == targets].sum() / weights.sum()
    pa_class = [weights[:, tau][signals[:, tau] == targets[:, tau]].sum()
                / weights[:, tau].sum() for tau in range(prices.shape[1] - 1)]

    return {'pa_total': pa_total, 'pa_class': pa_class}



