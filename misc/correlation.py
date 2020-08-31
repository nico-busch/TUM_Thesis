import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import adfuller
from sklearn.preprocessing import PowerTransformer

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

df = pd.read_csv('data/reuters.csv', parse_dates=['DATE'], index_col='DATE', dayfirst=True)

for x in df.columns:
    while True:
        p = adfuller(df[x].dropna())[1]
        if p <= 0.05:
            break
        df[x] = df[x].diff()


# df2 = df.drop(['SPOT', 'M1', 'M2', 'M3', 'M4', 'GASPOOL_SPOT', 'TTF_SPOT', 'TTF_M1'], axis=1)

# corr = pd.concat([df.corrwith(df['SPOT'].shift(-i)) for i in range(12)], axis=1)

corr = df.corr()

cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, center=0)
plt.show()
exit()

fig, axes = plt.subplots(figsize=(14/2.54, 3.1), ncols=4, nrows=2, sharey=True, sharex=True)
