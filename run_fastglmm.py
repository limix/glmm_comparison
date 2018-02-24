import numpy as np
import numpy_sugar as ns
from glimix_core.glmm import GLMMExpFam
from time import time

G = np.load('null_G.npy')
ntri = np.load('null_ntri.npy')
nsuc = np.load('null_nsuc.npy')
N, P = G.shape

QS = ns.linalg.economic_qs(G.dot(G.T))
X = np.ones((N, 1))

ntri = np.asarray(ntri, float)
nsuc = np.asarray(nsuc, float)

start = time()
glmm = GLMMExpFam((nsuc, ntri), "binomial", X, QS)
glmm.fit(verbose=True)
stop = time()
elapsed = stop - start
print("Elapsed: {}".format(elapsed))
np.save("out/fastglmm_N{}".format(N), elapsed)
