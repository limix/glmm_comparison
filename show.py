import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import numpy as np

SMALL_SIZE = 20
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

tick=2
ax = plt.gca()
ax.xaxis.set_tick_params(width=tick)
ax.yaxis.set_tick_params(width=tick)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(tick)


def pretty(m):
    if m == 'stan':
        return 'STAN'
    if m == 'fastglmm':
        return 'Binomial-FastGLMM'
    if m == 'macau':
        return 'MACAU'
    return m


methods = ['fastglmm', 'stan', 'macau']
data = dict(method=[], N=[], elapsed=[])
for m in methods:
    files = glob("out/{}_*.npy".format(m))
    for f in files:
        e = np.load(f).item()
        data['method'].append(m)
        data['N'].append(int(f[6+len(m):-4]))
        data['elapsed'].append(e)


df = pd.DataFrame(data=data)
methods = list(df['method'].unique())

for m in methods:
    df0 = df.query("method == '{}'".format(m))
    df0 = df0.sort_values(by='N')
    plt.plot(df0['N'], df0['elapsed'], label=pretty(m), linewidth=2)

plt.xlabel("sample size")
plt.ylabel("elapsed time (seconds)")
plt.legend()


plt.tight_layout()
# plt.show()
plt.savefig("performance.pdf")
