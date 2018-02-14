import sys
import numpy as np

if __name__ == '__main__':

    N = int(sys.argv[1])
    P = N

    random = np.random.RandomState(N)

    G = random.randn(N, P)
    G /= G.std(0)
    G -= G.mean()
    G /= np.sqrt(G.shape[1])
    np.save('G', G)

    u = 0.25 * random.randn(P)
    e = 0.75 * random.randn(N)
    z = 0.5 + G.dot(u) + e

    theta = 1/(1 + np.exp(-z))
    ntri = random.randint(10, 300, N)
    nsuc = random.binomial(ntri, theta)

    np.save('ntri', ntri)
    np.save('nsuc', nsuc)
