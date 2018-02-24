import numpy as np
import numpy_sugar as ns
from glimix_core.glmm import GLMMExpFam
from glimix_core.glmm import GLMMNormal
from time import time

G = np.load('G.npy')
SNP = np.load('X.npy')
ntri = np.load('ntri.npy')
nsuc = np.load('nsuc.npy')
N, P = G.shape

QS = ns.linalg.economic_qs(G.dot(G.T))
X = np.ones((N, 1))
S = SNP.shape[1]

ntri = np.asarray(ntri, float)
nsuc = np.asarray(nsuc, float)

start = time()
glmm = GLMMExpFam((nsuc, ntri), "binomial", X, QS)
glmm.fit(verbose=True)

eta = glmm.site.eta
tau = glmm.site.tau

gnormal = GLMMNormal(eta, tau, X, QS)
gnormal.fit(verbose=False)

flmm = gnormal.get_fast_scanner()
flmm.set_scale(1.0)
flmm.null_lml()

flmm.fast_scan(SNP, verbose=False)

stop = time()
elapsed = stop - start
print("Elapsed: {}".format(elapsed))
np.save("sca/fastglmm_S{}".format(S), elapsed)
