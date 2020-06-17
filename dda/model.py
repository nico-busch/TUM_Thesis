import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.model_selection import KFold


class DDA:

    def __init__(self, prices, features, demand, reg=None, big_m=False):

        self.prices = prices
        self.features = features
        self.demand = demand
        self.reg = reg
        self.big_m = big_m

        self.beta = None

    def train(self):

        if self.reg in ['l1', 'l2']:

            cv = KFold(n_splits=2)
            lambda1 = 0
            lambda2 = 0
            lambda_best = None
            c_best = np.inf
            beta = np.full([self.features.shape[1], self.prices.shape[1]], np.inf)
            count = 0

            while np.sum(beta[1:, 1:]) >= 10e-3 and count <= 1e3:

                c_split = np.empty(cv.get_n_splits())

                for i, (train, val) in enumerate(cv.split(self.prices)):

                    _, beta = self.optimize(self.prices[train], self.features[train], self.demand[train],
                                            lambda1, lambda2, self.big_m)
                    c_split[i] = self.prescribe(self.prices[val], self.features[val], self.demand[val], beta).mean()
                c = np.mean(c_split)

                if c < c_best:
                    c_best = c
                    lambda_best = lambda1, lambda2
                if self.reg == 'l1':
                    lambda1 += 0.01
                if self.reg == 'l2':
                    lambda2 += 0.01

                count += 1

        else:
            lambda_best = 0, 0

        obj, self.beta = self.optimize(self.prices, self.features, self.demand, *lambda_best, self.big_m)

        return obj

    @staticmethod
    def optimize(prices, features, demand, lambda1, lambda2, big_m):

        T = prices.shape[0]
        F = prices.shape[1] - 1
        N = features.shape[1] - 1

        p = {(t, tau): prices[t - 1, tau] for t in range(1, T + 1) for tau in range(F + 1)}
        X = {(i, t): features[t - 1, i] for t in range(1, T + 1) for i in range(N + 1)}
        d = {t: demand[t - 1] for t in range(1, T + 1)}

        m = gp.Model()
        m.Params.outputFlag = 0
        m.Params.timeLimit = 30

        q = m.addVars(range(1, T + 1), range(F + 1), vtype=GRB.BINARY)
        beta = m.addVars(range(N + 1), range(F + 1))
        w = m.addVars(range(1, N + 1), range(1, F + 1), vtype=GRB.BINARY)
        beta_abs = m.addVars(range(1, N + 1), range(1, F + 1))

        m.setObjective(gp.quicksum(p[t, tau] * d[t + tau] * q[t, tau]
                                   for t in range(1, T + 1) for tau in range(F + 1) if tau <= T - t) / T
                       + lambda1 * gp.quicksum(w[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1))
                       + lambda2 * gp.quicksum(beta_abs[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1)),
                       GRB.MINIMIZE)

        m.addConstrs(gp.quicksum(q[t - tau, tau] for tau in range(F + 1) if tau <= t - 1) == 1 for t in range(1, T + 1))
        m.addConstrs(beta_abs[i, tau] == gp.abs_(beta[i, tau]) for i in range(1, N + 1) for tau in range(1, F + 1))

        if big_m:
            M = 1e6
            m.addConstrs(-M * (1 - q[t, tau]) <= gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) - p[t, tau]
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs(
                M * (q[t, tau] + gp.quicksum(q[t + tau - a, a] for a in range(F + 1) if t + tau - 1 >= a > tau))
                >= gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) - p[t, tau]
                for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs(M * w[i, tau] >= beta[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1))
            m.addConstrs(-M * w[i, tau] <= beta[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1))

        else:
            q_aux = m.addVars(range(1, T + 1), range(F + 1), vtype=GRB.BINARY)
            m.addConstrs(q_aux[t, tau] * 2 >= q[t, tau]
                         + gp.quicksum(q[t + tau - a, a] for a in range(F + 1) if t + tau - 1 >= a > tau)
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs(q_aux[t, tau] <= q[t, tau]
                         + gp.quicksum(q[t + tau - a, a] for a in range(F + 1) if t + tau - 1 >= a > tau)
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs((q_aux[t, tau] == 0) >>
                         (gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) <= p[t, tau])
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs((q[t, tau] == 1) >> (gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) >= p[t, tau])
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs((w[i, tau] == 0) >> (beta[i, tau] <= 0) for i in range(1, N + 1) for tau in range(1, F + 1))
            m.addConstrs((w[i, tau] == 1) >> (beta[i, tau] >= 0) for i in range(1, N + 1) for tau in range(1, F + 1))

        m.optimize()
        beta = np.array([value.x for value in beta.values()]).reshape(N + 1, F + 1)

        return m.objVal, beta

    @staticmethod
    def prescribe(prices, features, demand, beta):

        thresholds = np.sum(features[:, :, np.newaxis] * beta[np.newaxis, :, :], axis=1)
        signals = (prices <= thresholds)

        satisfied = np.zeros(prices.shape[0], dtype=np.bool_)
        c = np.zeros(prices.shape[0])

        for t in range(prices.shape[0]):
            if not satisfied[t]:
                c[t] += prices[t, 0] * demand[t]

            for tau in range(1, prices.shape[1]):
                if tau <= prices.shape[0] - 1 - t:
                    if signals[t, tau] and not satisfied[t + tau]:
                        c[t] += prices[t, tau] * demand[t + tau]
                        satisfied[t + tau] = True

        return c
