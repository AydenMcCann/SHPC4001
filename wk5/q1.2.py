import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def f(x):
    return 1 - np.abs(4 * x / 1)


def a(k):
    return (4 * (1 - (-1) ** k)) / ((k ** 2) * (np.pi ** 2))


def b(k):
    return 0


def fourier_series(a_fn, b_fn, l, n_terms, x):
    """ Fourier Series Implementation where:
    a_fn = a_k
    b_fn = b_k
    l = the period
    k_terms = number of terms for fourier series
    x = xvalue at which to evaluate
    """
    dx = 0.001
    kvals = np.zeros(n_terms + 1)
    for k in range(1, n_terms):
        kvals[k] = a_fn(k) * np.cos((2 * np.pi * k * x) / l) + b_fn(k) * np.sin((2 * np.pi * k * x) / l)
    fval = np.sum(kvals)
    return fval


nterms = 200  # max number of terms to calc error for:

errors0 = np.empty(nterms)
errors125 = np.empty(nterms)
errorsxvals = np.empty(nterms)

for i in range(nterms):
    k_terms = 2 + 4 * i  # values chosen as envelope of worst errors as suggested
    errorsxvals[i] = k_terms
    errors0[i] = np.abs(f(0) - fourier_series(a, b, 1, k_terms, 0))
    errors125[i] = np.abs(f(0.125) - fourier_series(a, b, 1, k_terms, 0.125))

# plotting
plt.plot(errorsxvals, errors0, label="x=0")
plt.loglog(errorsxvals, errors125, label="x=0.125")
plt.legend(loc="upper right")
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.xlabel('x')  # axis labels
plt.ylabel('error')
plt.grid(True)
plt.show()

""" Finding error scaling """

print("At X=0 Error scales like ~ O(1/n^{})".format(-round(np.polyfit(np.log(errorsxvals), np.log(errors0), 1)[0])))

print("At X=0.125 Error scales like ~ O(1/n^{})".format(-round(np.polyfit(np.log(errorsxvals), np.log(errors125), 1)[0])))
