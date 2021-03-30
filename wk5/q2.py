import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def func(x):
    """ defines f(x) to perform DFT / IDFT on
    """
    return np.exp(np.cos(x))


def funcvals(deltax):
    """ Takes input from below to define x values and function values for given discretisation
    """
    x = np.linspace(0, 4 * np.pi, int(4 * np.pi / deltax))
    f = np.empty(int(4 * np.pi / deltax))
    f = func(x)
    return x, f


def naive_dft(f):
    """ Naive DFT implementation"""
    N = len(f)
    F = np.zeros(N, dtype=np.complex128)
    for m in range(N):
        for n in range(N):
            F[m] = F[m] + f[n] * np.exp(-2 * np.pi * 1j * m * n / N)
        pass
    return F


def naive_idft(F):
    """ Naive IDFT implementation"""
    N = len(F)
    f = np.zeros(N, dtype=np.complex128)
    for m in range(N):
        for n in range(N):
            f[n] = f[n] + (1 / N) * F[m] * np.exp(2 * np.pi * 1j * m * n / N)
        pass
    return f


"""Based on preference uncomment code following either 'define via delta-x' or 
    'define via steps' """

# # define via delta-x #
# num = 2
# deltax = np.array([1,2])
# def legend(i):
#     """ creates captions for plots"""
#     return 'using delta x = {}'.format(deltax[i])


# define via steps #
num = 2
steps = np.array([10, 20])
deltax = np.empty(num)
deltax = 4 * np.pi / steps
def legend(i):
    """ creates captions for plots"""
    return 'using {} steps'.format(steps[i])


arraysize = [num, int(4 * np.pi / min(deltax))]  # creates array sizes based on the highest numpy of values needed

x = np.empty(arraysize)  # empty arrays for x, f and results
f = np.empty(arraysize)
ff = np.empty(arraysize)

for i in range(num):  # iterates through list of deltax vals to generate results
    tempx, tempf = funcvals(deltax[i])
    x[i, :] = np.linspace(np.min(tempx), np.max(tempx), arraysize[1])  # linear interpolation to match array lengths
    f[i, :] = np.interp(x[i, :], tempx, tempf)  # linear interpolation to match array lengths
    ff[i, :] = np.real((naive_idft(naive_dft(f[i, :]))))  # idft(dft(fx)) calculation




# plotting #
exactxs = np.linspace(0, 4 * np.pi, 200)  # higher resolution to compare quality of transform
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
for i in range(num):
    plt.plot(x[i, :], np.abs(naive_dft(f[i, :])), label='|DFT| {}'.format(legend(i)))
plt.xlabel('m')  # axis labels
plt.legend(loc="upper right")
plt.show()






# plotting #
exactxs = np.linspace(0, 4 * np.pi, 200)  # higher resolution to compare quality of transform
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
for i in range(num):
    plt.plot(x[i, :], ff[i, :], label='idft(dft(fx)) {}'.format(legend(i)))
plt.plot(exactxs, func(exactxs), label='f(x)')
plt.xlabel('x')  # axis labels
plt.legend(loc="upper left")
plt.show()
