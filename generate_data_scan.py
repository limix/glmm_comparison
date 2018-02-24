import sys
import numpy as np

if __name__ == '__main__':

    N = int(sys.argv[1])
    S = int(sys.argv[2])
    P = N

    random = np.random.RandomState(N)

    G = random.randn(N, P)
    G /= G.std(0)
    G -= G.mean()
    G /= np.sqrt(G.shape[1])
    np.save('G', G)

    X = random.randn(N, S)
    X /= X.std(0)
    X -= X.mean()
    X /= np.sqrt(X.shape[1])
    np.save('X', X)

    idx = random.choice(S, 5, replace=False)
    SNP = X[:, idx]
    SNP /= SNP.std(0)
    SNP /= np.sqrt(SNP.shape[1])

    u = 0.20 * random.randn(P)
    e = 0.75 * random.randn(N)
    z = 0.5 + G.dot(u) + e + SNP.dot(0.05 * random.randn(SNP.shape[1]))

    theta = 1/(1 + np.exp(-z))
    ntri = random.randint(10, 300, N)
    nsuc = random.binomial(ntri, theta)

    np.save('ntri', ntri)
    np.save('nsuc', nsuc)
