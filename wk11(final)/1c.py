import warnings
from mpi4py import MPI
import numpy as np

warnings.filterwarnings("ignore")

n = 3
COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()


COMM.Barrier()

a_local = np.empty((n, size * n), dtype=np.float64)

if rank == 0:
    a_global = np.zeros((size * n, size * n), dtype=np.float64)

    for i in range(n * size):
        a_global[i, :] = i


else:
    a_global = None

COMM.Scatter([a_global, MPI.FLOAT], [a_local, MPI.FLOAT], root=0)
a_local += rank
# print(a_local, '\n', flush=True)


COMM.barrier()

if rank > 0:
    upper = np.empty(size * n, dtype=np.float64)
else:
    upper = "Pass"

if rank < size - 1:
    lower = np.empty(size * n, dtype=np.float64)
else:
    lower = "Pass"

if rank == 0:
    COMM.Send([a_local[-1, :], MPI.DOUBLE], dest=1)  # send lower

if rank == size - 1:
    COMM.Send([a_local[0, :], MPI.DOUBLE], dest=rank-1)  # send upper

if 0 < rank < size - 1:
    COMM.Send([a_local[0, :], MPI.DOUBLE], dest=rank - 1)  # send upper
    COMM.Recv([upper, MPI.DOUBLE], source=rank - 1) # receive upper
    COMM.Send([a_local[-1, :], MPI.DOUBLE], dest=rank + 1)  # send lower
    COMM.Recv([lower, MPI.DOUBLE], source=rank + 1)

if rank == size - 1:
    COMM.Recv([upper, MPI.DOUBLE], source=rank - 1)

if rank == 0:
    COMM.Recv([lower, MPI.DOUBLE], source=rank + 1)



for i in range(a_local.shape[0]):

    if lower == "Pass":
        pass
    else:
        a_local[i, :] += lower
    if upper == "Pass":
        pass
    else:
        a_local[i, :] += upper

print(a_local, '\n', flush = True)
""" note only the 1c print statements are uncommented"""
