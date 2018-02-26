import os
import sys
import pickle as pkl
from time import time
from multiprocessing import Pool

import numpy as np
import pystan

import limix


def print_params(params):
    print("offset: {}".format(params['offset']))
    if 'snp_effect' in params:
        print("snp_effect: {}".format(params['snp_effect']))
    print("sigma2_g: {}".format(params['vg']))
    print("sigma2_e: {}".format(params['ve']))
    print("lp: {}".format(params['lp']))
    print("h2: {}".format(params['h2']))


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


def load_stan_alt_model():
    if not os.path.exists('stan_alt.pickle'):
        stan_code = open("glmm_alt.stan").read()
        sm = pystan.StanModel(model_code=stan_code)
        with open('stan_alt.pickle', 'wb') as f:
            pkl.dump(sm, f, pkl.HIGHEST_PROTOCOL)
    else:
        with open('stan_alt.pickle', 'rb') as f:
            sm = pkl.load(f)
    return sm


def read_data():
    G = np.load('G.npy')
    Gcandidates = np.load('Gcandidates.npy')
    ntri = np.load('ntri.npy')
    nsuc = np.load('nsuc.npy')
    N, P = G.shape
    K = G.dot(G.T)
    K += np.eye(K.shape[0]) * 1e-6
    data = {
        'K': K,
        'N': N,
        'P': P,
        'ntri': ntri,
        'nsuc': nsuc,
        'Gcandidates': Gcandidates
    }
    return data


def extract_params_null(fit):
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


def extract_params_alt(fit):
    params = fit.get_posterior_mean()
    offset = params[-5, 0]
    snp_effect = params[-4, 0]
    sigma_g = params[-3, 0]
    sigma_e = params[-2, 0]
    lp = params[-1, 0]
    h2 = sigma_g**2 / (sigma_g**2 + sigma_e**2)
    return {
        'offset': offset,
        'snp_effect': snp_effect,
        'h2': h2,
        'lp': lp,
        'vg': sigma_g**2,
        've': sigma_e**2
    }


def get_null_lml():
    sm = load_stan_model()
    data = read_data()
    fit = sm.sampling(data=data, n_jobs=1)
    params = extract_params_null(fit)
    return params['lp']


def get_alt_lml(i):
    sm = load_stan_alt_model()
    data = read_data()
    X = data['Gcandidates']
    data['g'] = X[:, i]
    fit = sm.sampling(data=data, n_jobs=1)
    params = extract_params_null(fit)
    return params['lp']


def get_ncandidates():
    data = read_data()
    return data['Gcandidates'].shape[1]


def worker(i):
    if i == 0:
        return get_null_lml()
    return get_alt_lml(i - 1)


def main(nworkers):
    P = get_ncandidates()

    with Pool(processes=nworkers) as pool:
        lmls = pool.map(worker, range(P))

    null_lml = lmls[0]
    alt_lmls = lmls[1:]

    pv = limix.stats.lrt_pvalues(null_lml, alt_lmls)
    print(pv)
    # np.save("out/stan_N{}".format(N), elapsed)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        nworkers = int(sys.argv[1])
    else:
        nworkers = 1
    main(nworkers)
