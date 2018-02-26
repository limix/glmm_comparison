import sys

import numpy as np
from numpy import sqrt

if __name__ == '__main__':

    N = int(sys.argv[1])
    P = N
    h2 = 0.25
    c2 = 0.20
    offset = 0
    ncandidates = 10
    ncausals = 1

    random = np.random.RandomState()

    G = random.randn(N, P)
    G /= G.std(0)
    G -= G.mean(0)
    G /= sqrt(G.shape[1])
    # np.save('G', G)
    np.save('K', G.dot(G.T))

    X = random.randn(N, 2)
    X[:, 0] = 1
    np.save('X', X)

    Gcandidates = G[:, :ncandidates].copy()
    Gcandidates /= Gcandidates.std(0)
    Gcandidates /= sqrt(Gcandidates.shape[1])
    np.save('G', Gcandidates)
    # np.save('Gcandidates', Gcandidates)

    u = random.randn(P)
    u = sqrt(h2) * G.dot(u)

    e = random.randn(N)
    e = sqrt(1 - h2 - c2) * e
    causals = Gcandidates[:, :ncausals].copy()
    causals /= causals.std(0)
    causals /= sqrt(causals.shape[1])
    c = sqrt(c2) * causals.dot(random.randn(ncausals))

    print("var[c] {}".format(np.var(c)))
    print("var[u] {}".format(np.var(u)))
    print("var[e] {}".format(np.var(e)))
    z = offset + c + u + e

    theta = 1 / (1 + np.exp(-z))
    ntri = random.randint(100, 300, N)
    nsuc = random.binomial(ntri, theta)

    np.save('ntri', ntri)
    np.save('nsuc', nsuc)
