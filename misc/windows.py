import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

spot = pd.read_csv('data/data.csv', parse_dates=['DATE'], index_col='DATE', usecols=['DATE', 'SPOT'], dayfirst=True)
spot.index.freq = 'M'

plt.figure(figsize=(6, 6 / 16 * 9))
ax = plt.gca()

ax.plot(spot, marker='o', color=(0, 101/255, 189/255), linewidth=1, markersize=5)
input_window = plt.Rectangle((pd.Timestamp('2013-07-01', freq='M'),
                              spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].min() - 5),
                              pd.Timedelta(11, unit='M'),
                              spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].max()
                              - spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].min() + 5,
                              color=(156 / 255, 157 / 255, 159 / 255), fill=False)
ax.add_patch(input_window)
output_window = plt.Rectangle((pd.Timestamp('2014-07-01', freq='M'),
                              spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].min() - 5),
                              pd.Timedelta(5, unit='M'),
                              spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].max()
                              - spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].min() + 5,
                              color=(156/255, 157/255, 159/255), fill=False)
ax.add_patch(output_window)
ax.text(pd.Timestamp('2014-01-01', freq='M'),
        spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].to_numpy().min() - 3.5,
        'Input window', horizontalalignment='center')
ax.text(pd.Timestamp('2014-9-16', freq='M'),
        spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].to_numpy().min() - 3.5,
        'Output' + '\n' + 'window', horizontalalignment='center')
ax.set_xlim(pd.Timestamp('2012-12-15'), pd.Timestamp('2016-01-15'))
ax.set_ylim(spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].to_numpy().min() - 7.5,
            spot.loc[pd.Timestamp('2012-12-15'):pd.Timestamp('2016-01-15')].to_numpy().max() + 2.5)
plt.ylabel('EUR/MWh')
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.15)

plt.savefig('misc/windows.pgf')
