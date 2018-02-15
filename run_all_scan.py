import numpy as np
from os.path import exists
from subprocess import call

N = 1000
Ss = np.arange(5, 101)
for S in Ss:
    call(["python", "generate_data_scan.py", str(N), str(S)])

    if not exists("sca/macau_S{}.npy".format(S)):
        call(["python", "run_macau_scan.py"])

    if not exists("sca/fastglmm_S{}.npy".format(S)):
        call(["python", "run_fastglmm_scan.py"])
