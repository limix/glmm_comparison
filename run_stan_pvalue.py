import os
import pickle as pkl
from time import time

import numpy as np
import pystan

if not os.path.exists('stan.pickle'):
    stan_code = open("glmm.stan").read()
    sm = pystan.StanModel(model_code=stan_code)
    with open('stan.pickle', 'wb') as f:
        pkl.dump(sm, f, pkl.HIGHEST_PROTOCOL)
else:
    with open('stan.pickle', 'rb') as f:
        sm = pkl.load(f)

G = np.load('G.npy')
ntri = np.load('ntri.npy')
nsuc = np.load('nsuc.npy')
N, P = G.shape
K = G.dot(G.T)
K += np.eye(K.shape[0]) * 1e-7
data = {'K': K, 'N': N, 'P': P, 'ntri': ntri, 'nsuc': nsuc}

start = time()
fit = sm.sampling(data=data, iter=1000, chains=1)
stop = time()
elapsed = stop - start
print("Elapsed: {}".format(elapsed))

params = fit.get_posterior_mean()
offset = params[-4, 0]
sigma_g = params[-3, 0]
sigma_e = params[-2, 0]
lp = params[-1, 0]
h2 = sigma_g**2 / (sigma_g**2 + sigma_e**2)

print("offset: {}".format(offset))
print("sigma2_g: {}".format(sigma_g**2))
print("sigma2_e: {}".format(sigma_e**2))
print("lp: {}".format(lp))
print("h2: {}".format(h2))

# np.save("out/stan_N{}".format(N), elapsed)
# h2: 0.48234971105110414
#     0.02289635403732991
