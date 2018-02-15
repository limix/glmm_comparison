import numpy as np
import numpy_sugar as ns
from glimix_core.glmm import GLMMExpFam
from time import time

import limix_ext as lxt

G = np.load('null_G.npy')
ntri = np.load('null_ntri.npy')
nsuc = np.load('null_nsuc.npy')
N, P = G.shape

X = np.ones((N, 1))

ntri = np.asarray(ntri, float)
nsuc = np.asarray(nsuc, float)
K = G.dot(G.T)

start = time()
lxt.macau.qtl.binomial_scan(nsuc, ntri, X, G[:, :1], K)
stop = time()
elapsed = stop - start
print("Elapsed: {}".format(elapsed))
np.save("out/macau_N{}".format(N), elapsed)
