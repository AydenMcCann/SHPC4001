import warnings
from mpi4py import MPI
import numpy as np
warnings.filterwarnings("ignore")

COMM = MPI.COMM_WORLD

rank = COMM.Get_rank()

if rank == 0:
    send_array = np.array(2 * [rank], dtype=np.int)
else:
    recv_array = np.array(2 * [rank], dtype=np.int) # needed to initialise array

if rank == 0:

    COMM.Send([send_array, MPI.INT], dest=1)

else:

    COMM.Recv([recv_array, MPI.INT], source=0)  #source changed to 0

    print(recv_array)