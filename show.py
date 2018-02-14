from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import numpy as np


def pretty(m):
    if m == 'stan':
        return 'STAN'
    if m == 'fastglmm':
        return 'Binomial-FastGLMM'
    return m


methods = ['stan', 'fastglmm']
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
    plt.plot(df0['N'], df0['elapsed'], label=pretty(m))

plt.xlabel("sample size")
plt.ylabel("elapsed time (seconds)")
plt.legend()
# plt.show()
plt.savefig("performance.pdf")
