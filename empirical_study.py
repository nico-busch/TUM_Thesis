import numpy as np
import pandas as pd
import tests
import viz

def run():

    df = pd.read_csv('data/data.csv', parse_dates=['Date'], index_col='Date', dayfirst=True)
    prices = df[['Spot', 'M1', 'M2', 'M3', 'M4']].to_numpy()
    demand = df['Demand'].to_numpy()
    features = prices

    viz.forward_curve(prices)

    exit()

    train_size = 6
    test_size = 6

    c_nn = tests.test_prescriptive(prices, features, demand, test_size)

    c_0 = tests.c_tau(prices[train_size:], demand[train_size:], 0)
    c_1 = tests.c_tau(prices[train_size:], demand[train_size:], 1)

    c_pf = tests.c_pf(prices[train_size:], demand[train_size:])

    costs = np.array([c_0, c_1, c_nn])
    pe = (costs - c_pf) / c_pf * 100

    print(pe)


