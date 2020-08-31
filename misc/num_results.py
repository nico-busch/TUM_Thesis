import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


df = pd.read_pickle('results/numerical_study.pkl')
df = df[['p_spot', 'p_m1', 'ridge', 'lasso', 'mlp_seo', 'rnn_seo', 'lstm_seo', 'dda_ml1', 'dda_ml2', 'mlp_ieo', 'rnn_ieo', 'lstm_ieo']]
df.columns = ['P-SPOT', 'P-M1', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM', 'Ridge', 'LASSO', 'MLP', 'RNN', 'LSTM']

df.columns = pd.MultiIndex.from_arrays([['Baseline'] * 2 + ['SEO'] * 5 + ['IEO'] * 5, df.columns])

print(df.groupby(['process', 'non_linear', 'sigma', 'train_size']).mean().to_string())

exit()

test = df.loc[pd.IndexSlice['rw', True, 20, :], :]

results = test.describe()
results = results.round(2)
for x in results.idxmin(axis=1).iteritems():
    results.loc[x] = r'textbf{' + str(results.loc[x]) + '}'

print(results.to_latex())
