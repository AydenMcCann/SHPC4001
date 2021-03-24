import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl

# only uncomment the desired system


# # SYSTEM 1
# M = np.diag(np.full(100, 1))
# K = (np.diag(np.full(100, 2)) + np.diag(np.ones(99), 1) + np.diag(np.ones(99), -1))

# # SYSTEM 2
# M = np.diag(np.full(100, 1))
# K = (np.diag(np.full(100, 2)) + np.diag(np.ones(99), 1) + np.diag(np.ones(99), -1))
# K[99,99] = 1

# # SYSTEM 3
# M = np.diag(np.full(100, 1))
# K = np.zeros([100, 100])
# for i in range(100):
#     K[i, i] = 1 + (i + 1) / 100 + 1 + (i + 2) / 100
# for i in range(99):
#     K[i + 1, i] = - (1 + (i + 2) / 100)
#     K[i, i + 1] = - (1 + (i + 2) / 100)



values, vectors = la.eig(M, K)  # solving for eigenvalues and eigenvectors


def sort_eig(eval, evec):
    """sorts the eigenvalues and eigenvectors from scipy.linalg.eig
    """
    eval = np.real_if_close(eval)
    idx = np.argsort(eval)
    evaly = eval[idx]
    evecy = evec.T[idx].T
    return evaly, evecy


evals, evecs = sort_eig(values, vectors) # output sorted eigenvalues and eigenvectors


def eom(y, omega, t):
    """takes inputs of y omega and t and outputs x given equation of motion in assignment
    """
    x = y * ((np.cos(omega * t) + j * np.sin(omega * t)))  # eulers formula
    return x


# inputs
modes = 5
time = 50
smoothness_factor = 10          # introduced to increase the smoothness of the curves for small t,
                                # essentially number of points inbetween integer values of t

time = time*smoothness_factor

positions = np.zeros([modes, time])
for k in range(modes):
    for j in range(time):
        positions[k:, j] = eom(vectors[k, 0], np.sqrt(values[k]),
                               j * (1/smoothness_factor))  # calculating position using eom given time and mode

# plotting
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
for k in range(modes):
    plt.plot(np.linspace(0, time * (1/smoothness_factor), len(vectors[:, k])), vectors[:, k])
plt.xlabel('t')  # axis labels
plt.ylabel('position')
plt.show()  # show plot
