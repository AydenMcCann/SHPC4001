import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def bisect_root(f, lower, upper, eps, max_iter=100):
    """Find zero of a function in an interval to a given precision using the bis
   ection method.
    Arguments:
    f: function, assumed continuous over interval [lower, upper]
    lower: lower bound on root
    upper: upper bound on root
    eps: target precision (size of final interval)
    max_iter: limit on number of iterations before exiting
    Returns:
    (root, step_count)
    """
    a, b = lower, upper
    fa, fb = f(a), f(b)
    if not (a < b and fa * fb < 0):
        raise ValueError('invalid input parameters or function')
    if f(a) == 0:
        return a, 0
    if f(b) == 0:
        return b, 0
    n = 0
    while b - a > eps:
        n += 1
        c = (a + b) / 2
        fc = f(c)
        if fc == 0:
            return c, n
        if fa * fc > 0:
            a, fa = c, fc
        else:
            b, fb = c, fc
    # return (a + b) / 2 , n
    return n

def secant_method(f, lower, upper, eps, max_iter=100):
    """Find zero of a function in an interval to a given precision using the bis
   ection method.
    Arguments:
    f: function, assumed continuous over interval [lower, upper]
    lower: lower bound on root
    upper: upper bound on root
    eps: target precision (size of final interval)
    max_iter: limit on number of iterations before exiting
    Returns:
    (root, step_count)
    """
    a, b = lower, upper
    fa, fb = f(a), f(b)
    if not (a < b and fa * fb < 0):
        raise ValueError('invalid input parameters or function')
    if f(a) == 0:
        return a, 0
    if f(b) == 0:
        return b, 0
    c = b - f(b) * ((b - a) / (f(b) - f(a)))
    n = 0
    while np.abs(c - b) > eps and n < max_iter:
        b2 = b
        b = b2
        b = c
        a = b2
        c = b - f(b) * ((b - a) / (f(b) - f(a)))
        n += 1
    return n


def function(x):
    return np.sin(x) + (np.cos(2 * x)) ** 3 - x


#       Plot to verify root was accurate       #
# steps = 1000
# xval = np.empty(steps)
# xval[0] = 0
# dt = 0.001
# array = np.empty(steps)
# for i in range(steps):
#     xval[i] = i * dt
#     array[i] = function(xval[i])
#
# plt.scatter(xval, array, s=2)
# plt.axvline(x=bisect_root(function, 0, 1, 0.01, 50), color="red")
# plt.axhline(0, color='red')
# plt.show()  # show plot

steps = 10000
xval = np.empty(steps)
xval[0] = 0
dt = 0.0001
a0 = 0
b0 = 1
bisectarray = np.empty(steps)
secantarray = np.empty(steps)
convergence = np.empty(steps)
for i in range(steps):
    xval[i] = 1 - i * dt
    bisectarray[i] = (bisect_root(function, 0, 1, xval[i], 500))
    secantarray[i] = (secant_method(function, 0, 1, xval[i], 500))
#    convergence[i] = (np.log(b0 - a0) - np.log(xval[i])) / (np.log(2))

mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.scatter(xval, bisectarray, s=2, label="Bisection Method Numerical Convergence")
plt.scatter(xval, secantarray, s=2, label="Secant Method Numerical Convergence")
#plt.scatter(xval, convergence, s=2, label="Bisection Method Analytical Convergence")
plt.xlabel('t (target precision)')  # axis labels
plt.ylabel('steps needed')
plt.legend(loc="upper right")

plt.show()  # show plot
