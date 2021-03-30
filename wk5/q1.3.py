import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 2 * x


def a(k):
    return 0


def b(k):
    return (2 * ((-1) ** (k + 1))) / (k * np.pi)


def fourier_series(a_fn, b_fn, l, n_terms, k_terms):
    """ Fourier Series Implementation where:
    a_fn = a_k
    b_fn = b_k
    l = the period
    n_terms = number of terms for sinc correction
    k_terms = number of terms for fourier series
    """
    dx = 0.001
    arraylen = int(l / dx) + 1
    kvals = np.zeros(k_terms + 1)
    fvals = np.zeros(arraylen)
    xvals = np.zeros(arraylen)
    exact = np.zeros(arraylen)
    for i in range(arraylen):
        xvals[i] = ((-l / 2) + i * dx)
        for k in range(1, k_terms):
            kvals[k] = (np.sinc(k / (n_terms + 1))) * (
                        a_fn(k) * np.cos((2 * np.pi * k * xvals[i]) / l) + b_fn(k) * np.sin(
                    (2 * np.pi * k * xvals[i]) / l))
        fvals[i] = np.sum(kvals)
    return fvals, xvals, exact


fvals_5, xvals, exact = fourier_series(a, b, 1, 5, 20)
fvals_10 = fourier_series(a, b, 1, 10, 20)[0]
fvals_20 = fourier_series(a, b, 1, 20, 20)[0]

exact = f(xvals)


def loop(fvals):  # stacks array horizontally 2 times (for plotting purposes)
    loop = np.hstack((fvals, fvals))
    return loop


loop_5 = loop(fvals_5)
loop_10 = loop(fvals_10)
loop_20 = loop(fvals_20)
loopexact = loop(exact)




# plotting
xval = np.linspace(-0.5, 3.5, len(loop_5))
plt.plot(xval, loop_5, label="n = 5")
plt.plot(xval, loop_10, label="n = 10")
plt.plot(xval, loop_20, label="n = 20")
plt.plot(xval, loopexact, label='f(x)')
plt.legend(loc="upper left")
plt.show()

