import pandas as pd

# Prepare dataframes
df = pd.read_pickle('results/numerical_study.pkl')
df = df[['p_spot', 'p_m1', 'lasso', 'ridge', 'mlp_seo', 'rnn_seo', 'lstm_seo', 'dda_ml1', 'dda_ml2', 'mlp_ieo', 'rnn_ieo', 'lstm_ieo']]
df.columns = ['P-SPOT', 'P-M1', 'LASSO', 'Ridge', 'MLP', 'RNN', 'LSTM', 'LASSO', 'Ridge', 'MLP', 'RNN', 'LSTM']
df.columns = pd.MultiIndex.from_arrays([['Baseline'] * 2 + ['Sequential'] * 5 + ['Integrated'] * 5, df.columns])

# Print to latex format
results = df.groupby(['process', 'non_linear', 'sigma', 'train_size']).describe().stack(-1)
results = results.drop('count', axis=0, level=4)
results = results.reindex(['rw', 'mr'], level=0)
results = results.reindex([False, True], level=1)
results = results.reindex(['mean', 'min', '25%', '50%', '75%', 'max'], level=4)
results = results.reindex(['Baseline', 'SEO', 'IEO'], axis=1, level=0)
results = pd.concat([results.reindex(['P-SPOT', 'P-M1'], axis=1, level=1),
                     results.reindex(['LASSO', 'Ridge', 'MLP', 'RNN', 'LSTM'], axis=1, level=1)], axis=1)
results.index = results.index.set_levels(['RW', 'MR'], level=0)
results.index = results.index.set_levels(['Linear', 'Logarithmic'], level=1)
results.index = results.index.set_levels(['$sigma = 10$', '$sigma = 20$'], level=2)
results.index = results.index.set_levels(['$prime = 24$', '$prime = 72$'], level=3)
results.index = results.index.set_levels(['Mean', 'Min', '25%', 'Median', '75%', 'Max'], level=4)
results = results.round(2)
print(results.to_latex(multirow=True, longtable=True, float_format='%.2f'))
