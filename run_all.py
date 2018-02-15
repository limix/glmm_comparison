import numpy as np
from os.path import exists
from subprocess import call

# Ns = np.asarray(np.linspace(936, 5000, 100).round(), int)
Ns = np.asarray(np.linspace(100, 1000, 100).round(), int)
for N in Ns:
    call(["python", "generate_data.py", str(N)])

    if not exists("out/macau_N{}.npy".format(N)):
        call(["python", "run_macau.py", str(N)])

    if not exists("out/stan_N{}.npy".format(N)):
        call(["python", "run_stan.py", str(N)])

    if not exists("out/fastglmm_N{}.npy".format(N)):
        call(["python", "run_fastglmm.py", str(N)])
