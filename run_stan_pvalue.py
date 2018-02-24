import os
import pickle as pkl
from time import time

import numpy as np
import pystan


class Time(object):
    def __init__(self):
        self._start = None

    def __enter__(self):
        self._start = time()
        return self

    def __exit__(self, *_):
        self._stop = time()

    @property
    def elapsed(self):
        return self._stop - self._start


def load_stan_model():
    if not os.path.exists('stan.pickle'):
        stan_code = open("glmm.stan").read()
        sm = pystan.StanModel(model_code=stan_code)
        with open('stan.pickle', 'wb') as f:
            pkl.dump(sm, f, pkl.HIGHEST_PROTOCOL)
    else:
        with open('stan.pickle', 'rb') as f:
            sm = pkl.load(f)
    return sm


def read_data():
    G = np.load('G.npy')
    ntri = np.load('ntri.npy')
    nsuc = np.load('nsuc.npy')
    N, P = G.shape
    K = G.dot(G.T)
    K += np.eye(K.shape[0]) * 1e-7
    data = {'K': K, 'N': N, 'P': P, 'ntri': ntri, 'nsuc': nsuc}
    return data


def extract_params(fit):
    params = fit.get_posterior_mean()
    offset = params[-4, 0]
    sigma_g = params[-3, 0]
    sigma_e = params[-2, 0]
    lp = params[-1, 0]
    h2 = sigma_g**2 / (sigma_g**2 + sigma_e**2)
    return {
        'offset': offset,
        'h2': h2,
        'lp': lp,
        'vg': sigma_g**2,
        've': sigma_e**2
    }


sm = load_stan_model()
data = read_data()

with Time() as t:
    fit = sm.sampling(data=data, iter=1000, chains=1)

print("Elapsed: {}".format(t.elapsed))

params = extract_params(fit)

print("offset: {}".format(params['offset']))
print("sigma2_g: {}".format(params['vg']))
print("sigma2_e: {}".format(params['ve']))
print("lp: {}".format(params['lp']))
print("h2: {}".format(params['h2']))

# np.save("out/stan_N{}".format(N), elapsed)
# h2: 0.48234971105110414
#     0.02289635403732991
#     0.027030161220725237
#     0.1432504998959289
#     0.30640576826433913
