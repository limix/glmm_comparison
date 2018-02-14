import numpy as np
from subprocess import call

Ns = np.asarray(np.linspace(100, 600, 10).round(), int)
for N in Ns:
    call(["python", "generate_data.py", str(N)])
    call(["python", "run_stan.py", str(N)])
    call(["python", "run_fastglmm.py", str(N)])
