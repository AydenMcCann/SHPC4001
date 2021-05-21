import warnings
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

warnings.filterwarnings("ignore")

COMM = MPI.COMM_WORLD
rank = COMM.Get_rank()
size = COMM.Get_size()

num_points = 60
n = int(num_points / size)

COMM.Barrier()

if rank == 0:
    m = np.zeros((num_points, num_points), dtype=float)
    pi_c = np.pi
    x = np.linspace(0, pi_c, num_points)
    m[0, :] = np.sin(x)
    m[num_points - 1, :] = np.sin(x)

else:
    m = None


def Jacobi(m):
    m_local = np.zeros((n, num_points), dtype=np.float64)
    m_local_update = np.zeros((n, num_points), dtype=np.float64)
    m_local_error = np.zeros((num_points), dtype=np.float64)
    COMM.Scatter([m, MPI.FLOAT], [m_local, MPI.FLOAT], root=0)
    COMM.Scatter([m, MPI.FLOAT], [m_local_update, MPI.FLOAT], root=0)
    COMM.barrier()
    errors = np.empty(size)

    if rank > 0:
        upper = np.empty(size * n, dtype=np.float64)

    if rank < size - 1:
        lower = np.empty(size * n, dtype=np.float64)

    count = 0
    error = 100

    while error > 0.001:

        if rank == 0:
            COMM.Send([m_local[-1, :], MPI.DOUBLE], dest=1)  # send lower

        if rank == size - 1:
            COMM.Send([m_local[0, :], MPI.DOUBLE], dest=rank - 1)  # send upper

        if 0 < rank < size - 1:
            COMM.Send([m_local[0, :], MPI.DOUBLE], dest=rank - 1)  # send upper
            COMM.Recv([upper, MPI.DOUBLE], source=rank - 1)  # receive upper
            COMM.Send([m_local[-1, :], MPI.DOUBLE], dest=rank + 1)  # send lower
            COMM.Recv([lower, MPI.DOUBLE], source=rank + 1)

        if rank == size - 1:
            COMM.Recv([upper, MPI.DOUBLE], source=rank - 1)

        if rank == 0:
            COMM.Recv([lower, MPI.DOUBLE], source=rank + 1)

        COMM.barrier()

        if rank == size - 1:
            for i in range(m_local.shape[0] - 1):
                for j in range(1, m_local.shape[1] - 1):
                    if i == 0:
                        m_local_update[i, j] = (upper[j] + m_local[i, j + 1] + m_local[i, j - 1] + m_local[
                            i + 1, j]) / 4


                    else:
                        m_local_update[i, j] = (m_local[i - 1, j] + m_local[i, j + 1] + m_local[i, j - 1] + m_local[
                            i + 1, j]) / 4

        if 0 < rank < size - 1:
            for i in range(m_local.shape[0]):
                for j in range(1, m_local.shape[1] - 1):
                    if i == m_local.shape[0] - 1:
                        m_local_update[i, j] = (m_local[i - 1, j] + m_local[i, j + 1] + m_local[i, j - 1] + lower[
                            j]) / 4

                    if i == 0:
                        m_local_update[i, j] = (upper[j] + m_local[i, j + 1] + m_local[i, j - 1] + m_local[
                            i + 1, j]) / 4

                    if 0 < i < m_local.shape[0] - 1:
                        m_local_update[i, j] = (m_local[i - 1, j] + m_local[i, j + 1] + m_local[i, j - 1] + m_local[
                            i + 1, j]) / 4

                    else:
                        pass

        if rank == 0:
            for i in range(1, m_local.shape[0]):
                for j in range(1, m_local.shape[1] - 1):
                    if i == m_local.shape[0] - 1:
                        m_local_update[i, j] = (m_local[i - 1, j] + m_local[i, j + 1] + m_local[i, j - 1] + lower[
                            j]) / 4

                    else:
                        m_local_update[i, j] = (m_local[i - 1, j] + m_local[i, j + 1] + m_local[i, j - 1] + m_local[
                            i + 1, j]) / 4

        for i in range(0, m_local.shape[0]):
            m_local_error[i] = np.sum(np.abs(m_local_update[i][:] - m_local[i][:]))

        count = count + 1
        error_local = np.max(m_local_error)
        COMM.Gather([error_local, MPI.DOUBLE], [errors, MPI.DOUBLE], root=0)
        error = COMM.bcast(np.max(errors), root=0)
        COMM.barrier()
        m_local[:] = m_local_update[:]
        COMM.barrier()

    COMM.barrier()

    m_total = np.empty([size, n, num_points])
    COMM.Gather([m_local, MPI.DOUBLE], [m_total, MPI.DOUBLE], root=0)
    m_total = COMM.bcast(m_total, root=0)
    m = np.reshape(m_total, [num_points, num_points])

    return m, count, error


tic = time.perf_counter()  # begin timer
m, count, error = Jacobi(m)
toc = time.perf_counter()

time = toc - tic
if rank == 0:
    plt.matshow(m)
    plt.savefig('output.png')
    file_object = open('output.txt', 'a')
    file_object.write("******************* \n")
    file_object.write("number of processes = {} \n".format(size))
    file_object.write("finished after {} iterations \n".format(count))
    file_object.write("error = {} \n".format(error))
    file_object.write("function run time = {} \n".format(time))
