from numerical_study import simulation

simulation()

# df = pd.read_csv('data.csv', parse_dates=['Date'], index_col='Date', dayfirst=True)
# spot = df['Spot'].to_numpy()
# forward = df[['M1', 'M2']].to_numpy()
# demand = df['Demand'].to_numpy()
#
# n_hidden = 100
# n_steps = 5
# n_forwards = forward.shape[1]
# n_input = forward.shape[1] + 1
#
# model = NN(n_steps, n_input, n_forwards, n_hidden)
#
# trainer = Trainer(model, spot, forward, demand, None)
#
# trainer.train()
