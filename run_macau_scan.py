import numpy as np
import numpy_sugar as ns
from glimix_core.glmm import GLMMExpFam
from time import time

import limix_ext as lxt

G = np.load('G.npy')
SNP = np.load('X.npy')
ntri = np.load('ntri.npy')
nsuc = np.load('nsuc.npy')
N, P = G.shape

X = np.ones((N, 1))
S = SNP.shape[1]

ntri = np.asarray(ntri, float)
nsuc = np.asarray(nsuc, float)
K = G.dot(G.T)

start = time()
lxt.macau.qtl.binomial_scan(nsuc, ntri, X, SNP, K)
stop = time()
elapsed = stop - start
print("Elapsed: {}".format(elapsed))
np.save("sca/macau_S{}".format(S), elapsed)
