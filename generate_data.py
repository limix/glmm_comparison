import sys

import numpy as np
from numpy import sqrt

if __name__ == '__main__':

    N = int(sys.argv[1])
    P = N

    random = np.random.RandomState()

    G = random.randn(N, P)
    G /= G.std(0)
    G -= G.mean(0)
    G /= sqrt(G.shape[1])
    np.save('G', G)

    h2 = 0.25

    u = random.randn(P)
    u = sqrt(h2) * G.dot(u)

    e = random.randn(N)
    e = sqrt(1 - h2) * e

    print("var[u] {}".format(np.var(u)))
    print("var[e] {}".format(np.var(e)))
    z = 0.0 + u + e

    theta = 1 / (1 + np.exp(-z))
    ntri = random.randint(100, 300, N)
    # ntri = random.randint(10, 300, N)
    nsuc = random.binomial(ntri, theta)

    np.save('null_ntri', ntri)
    np.save('null_nsuc', nsuc)
