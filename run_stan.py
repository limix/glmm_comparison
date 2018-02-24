import pystan
import os
import numpy as np
from time import time
import pickle as pkl

if not os.path.exists('stan.pickle'):
    stan_code = open("glmm.stan").read()
    sm = pystan.StanModel(model_code=stan_code)
    with open('stan.pickle', 'wb') as f:
        pkl.dump(sm, f, pkl.HIGHEST_PROTOCOL)
else:
    with open('stan.pickle', 'rb') as f:
        sm = pkl.load(f)

G = np.load('null_G.npy')
ntri = np.load('null_ntri.npy')
nsuc = np.load('null_nsuc.npy')
N, P = G.shape
data = {'G': G, 'N': N, 'P': P, 'ntri': ntri, 'nsuc': nsuc}

start = time()
fit = sm.sampling(data=data, iter=1000, chains=4, n_jobs=1)
stop = time()
elapsed = stop - start
print("Elapsed: {}".format(elapsed))
np.save("out/stan_N{}".format(N), elapsed)
