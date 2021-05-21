import warnings
from mpi4py import MPI
import numpy as np

warnings.filterwarnings("ignore")

n = 3
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

a_local = np.empty((n, size * n), dtype=np.int)

if rank == 0:
    a_global = np.zeros((size * n, size * n), dtype=np.int)

    for i in range(n * size):
        a_global[i, :] = i
    # print(a_global)

else:
    a_global = None

COMM.Scatter([a_global, MPI.INT], [a_local, MPI.INT], root=0)
a_local += rank
print(a_local, '\n', flush=True)
