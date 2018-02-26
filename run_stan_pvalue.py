import os
import sys
import pickle as pkl
from time import time
from multiprocessing import Pool

import numpy as np
import pystan

import limix


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


def load_model():
    if not os.path.exists('glmm.pickle'):
        stan_code = open("glmm.stan").read()
        sm = pystan.StanModel(model_code=stan_code)
        with open('glmm.pickle', 'wb') as f:
            pkl.dump(sm, f, pkl.HIGHEST_PROTOCOL)
    else:
        with open('glmm.pickle', 'rb') as f:
            sm = pkl.load(f)
    return sm


def extract_params(N, P, fit):
    nitems = dict(effsiz=P, u_effsiz=N, e=N, u=N)
    pars = fit.model_pars
    for k in pars:
        if k not in nitems:
            nitems[k] = 1

    params = fit.get_posterior_mean()
    i = 0
    r = dict()
    for k in pars:
        r[k] = np.mean(params[i:i+nitems[k], :], 1)
        i += nitems[k]

    r['lp'] = np.mean(params[-1, :])
    for k in ['lp', 'sigma_g', 'sigma_e']:
        r[k] = r[k].item()

    for k in ['u', 'u_effsiz', 'e']:
        if k in r:
            del r[k]

    return r


def scan(nsuc, ntri, X, G, K):
    sm = load_model()
    N = K.shape[0]
    P = X.shape[1]
    data = dict(N=N, P=P, nsuc=nsuc, ntri=ntri, X=X, K=K)

    fit = sm.sampling(data=data)
    params = extract_params(N, P, fit)
    null_lml = params['lp']

    alt_lmls = []
    for i in range(G.shape[1]):
        data['X'] = np.concatenate((X, G[:, i:i+1]), axis=1)
        data['P'] = data['X'].shape[1]
        fit = sm.sampling(data=data)
        params = extract_params(N, P, fit)
        alt_lmls.append(params['lp'])

    pv = limix.stats.lrt_pvalues(null_lml, alt_lmls)
    return pv


if __name__ == '__main__':
    nsuc = np.load('nsuc.npy')
    ntri = np.load('ntri.npy')
    X = np.load('X.npy')
    G = np.load('G.npy')
    K = np.load('K.npy')
    pv = scan(nsuc, ntri, X, G, K)
    print(pv)
    # if len(sys.argv) == 2:
    #     nworkers = int(sys.argv[1])
    # else:
    #     nworkers = 1
    # main(nworkers)
