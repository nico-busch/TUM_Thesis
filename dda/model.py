import timeit
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class DDA:

    def __init__(self, prices, features, demand, reg=None, big_m=False):

        self.prices = prices
        self.features = features
        self.demand = demand
        self.reg = reg
        self.big_m = big_m

        # Normalize features
        self.scaler = StandardScaler()
        self.features_std = self.scaler.fit_transform(features)

        self.beta = None

    def train(self):

        start = timeit.default_timer()

        if self.reg in ['lasso', 'ridge']:

            cv = KFold(n_splits=2)
            lam = 0
            lam_best = None
            c_best = np.inf
            all_zero = False

            while not all_zero and timeit.default_timer() - start <= 30:

                # Cross-validate
                c_split = np.empty(cv.get_n_splits())
                for i, (train, val) in enumerate(cv.split(self.prices)):

                    _, beta = self.optimize(self.prices[train], self.features_std[train], self.demand[train],
                                            self.big_m, lam)
                    if np.sum(beta[1:, 1:]) <= 10e-3:
                        all_zero = True
                    c_split[i] = self.prescribe(self.prices[val], self.features_std[val], self.demand[val], beta).mean()
                c = np.mean(c_split)

                if c < c_best:
                    c_best = c
                    lam_best = lam

                lam += 0.01

            obj, self.beta = self.optimize(self.prices, self.features_std, self.demand, self.big_m, lam_best)

        else:
            obj, self.beta = self.optimize(self.prices, self.features_std, self.demand, self.big_m)

        return obj

    def optimize(self, prices, features, demand, big_m, lam=0):

        # Add intercept
        features = np.hstack([np.ones([features.shape[0], 1]), features])

        # Prepare indices
        T = prices.shape[0]
        F = prices.shape[1] - 1
        N = features.shape[1] - 1

        # Prepare data
        M = 1e6
        p = {(t, tau): prices[t - 1, tau] for t in range(1, T + 1) for tau in range(F + 1)}
        X = {(i, t): features[t - 1, i] for t in range(1, T + 1) for i in range(N + 1)}
        d = {t: demand[t - 1] for t in range(1, T + 1)}

        # Build model
        m = gp.Model()
        m.Params.outputFlag = 0
        m.Params.timeLimit = 30

        # Add decision variables
        q = m.addVars(range(1, T + 1), range(F + 1), vtype=GRB.BINARY)
        beta = m.addVars(range(N + 1), range(F + 1))

        # Build objective function
        obj = gp.quicksum(p[t, tau] * d[t + tau] * q[t, tau] for t in range(1, T + 1)
                          for tau in range(F + 1) if tau <= T - t) / T

        # Add constraints
        m.addConstrs(gp.quicksum(q[t - tau, tau] for tau in range(F + 1) if tau <= t - 1) == 1 for t in range(1, T + 1))
        if big_m:
            m.addConstrs(-M * (1 - q[t, tau]) <= gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) - p[t, tau]
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs(
                M * (q[t, tau] + gp.quicksum(q[t + tau - a, a] for a in range(F + 1) if t + tau - 1 >= a > tau))
                >= gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) - p[t, tau]
                for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
        else:
            aux = m.addVars(range(1, T + 1), range(F + 1), vtype=GRB.BINARY)
            m.addConstrs((aux[t, tau] == 0) >>
                         (gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) <= p[t, tau])
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs((aux[t, tau] == 1) >>
                         (gp.quicksum(beta[i, tau] * X[i, t] for i in range(N + 1)) >= p[t, tau])
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs(q[t, tau] <= aux[t, tau] for tau in range(1, F + 1) for t in range(1, T + 1 - tau))
            m.addConstrs(q[t, tau] >= aux[t, tau]
                         - gp.quicksum(q[t + tau - a, a] for a in range(F + 1) if t + tau - 1 >= a > tau)
                         for tau in range(1, F + 1) for t in range(1, T + 1 - tau))

        # Add LASSO regularization
        if self.reg == 'lasso':
            w = m.addVars(range(1, N + 1), range(1, F + 1), vtype=GRB.BINARY)
            obj += lam * gp.quicksum(w[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1))
            if big_m:
                m.addConstrs(M * w[i, tau] >= beta[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1))
                m.addConstrs(-M * w[i, tau] <= beta[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1))
            else:
                m.addConstrs((w[i, tau] == 0) >> (beta[i, tau] <= 0)
                             for i in range(1, N + 1) for tau in range(1, F + 1))
                m.addConstrs((w[i, tau] == 1) >> (beta[i, tau] >= 0)
                             for i in range(1, N + 1) for tau in range(1, F + 1))

        # Add ridge regularization
        if self.reg == 'ridge':
            beta_abs = m.addVars(range(1, N + 1), range(1, F + 1))
            obj += lam * gp.quicksum(beta_abs[i, tau] for i in range(1, N + 1) for tau in range(1, F + 1))
            m.addConstrs(beta_abs[i, tau] == gp.abs_(beta[i, tau]) for i in range(1, N + 1) for tau in range(1, F + 1))

        # Generate output
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()
        beta = np.array([value.x for value in beta.values()]).reshape(N + 1, F + 1)

        return m.objVal, beta

    def prescribe(self, prices, features, demand, beta=None):

        if beta is None:
            beta = self.beta

        features = np.hstack([np.ones([features.shape[0], 1]), features])

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
