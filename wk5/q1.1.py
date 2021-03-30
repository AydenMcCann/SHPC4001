import numpy as np
import matplotlib.pyplot as plt

""" Uncomment the desired wave type"""

# # Triangle Wave
# def f(x):
#     return 1 - np.abs(4 * x / 1)
#
#
# def a(k):
#     return (4 * (1 - (-1) ** k)) / ((k ** 2) * (np.pi ** 2))
#
#
# def b(k):
#     return 0




# Sawtooth Wave
def f(x):
    return 2*x


def a(k):
    return 0


def b(k):
    return (2 * ((-1) ** (k + 1))) / (k * np.pi)


def fourier_series(a_fn, b_fn, l, n_terms):
    """ Fourier Series Implementation where:
    a_fn = a_k
    b_fn = b_k
    l = the period
    n = number of terms
    """
    k_terms = n_terms
    dx = 0.001
    arraylen = int(l / dx) + 1
    kvals = np.zeros(k_terms + 1)
    fvals = np.zeros(arraylen)
    xvals = np.zeros(arraylen)
    exact = np.zeros(arraylen)
    for i in range(arraylen):
        xvals[i] = ((-l / 2) + i * dx)
        for k in range(1, k_terms):
            kvals[k] = (a_fn(k) * np.cos((2 * np.pi * k * xvals[i]) / l) + b_fn(k) * np.sin(
                (2 * np.pi * k * xvals[i]) / l))
        fvals[i] = np.sum(kvals)
    return fvals, xvals, exact


fvals_5, xvals, exact = fourier_series(a, b, 1, 5)
fvals_10 = fourier_series(a, b, 1, 10)[0]
fvals_20 = fourier_series(a, b, 1, 20)[0]

exact = f(xvals)



# sums
plt.title('Partial Sums')
plt.plot(xvals,fvals_5,label = "n = 5")
plt.plot(xvals,fvals_10,label = "n = 10")
plt.plot(xvals,fvals_20,label = "n = 20")
plt.plot(xvals,exact,label='f(x)')
plt.legend(loc="upper left")
plt.show()

# difference between sums
plt.title('Differences')
plt.plot(xvals,(exact-fvals_5),label = "n = 5")
plt.plot(xvals,(exact-fvals_10),label = "n = 10")
plt.plot(xvals,(exact-fvals_20),label = "n = 20")
plt.legend(loc="upper left")
plt.show()# difference between sums
