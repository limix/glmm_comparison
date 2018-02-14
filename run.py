import pystan
import numpy as np
import pdb

# pdb.set_trace()
stan_code = open("glmm.stan").read()
sm = pystan.StanModel(model_code=stan_code)

G = np.load('G.npy')
ntri = np.load('ntri.npy')
nsuc = np.load('nsuc.npy')
N, P = G.shape
data = {'G': G, 'N': N, 'P': P, 'ntri': ntri, 'nsuc': nsuc}
fit = sm.sampling(data=data, iter=1000, chains=4)
