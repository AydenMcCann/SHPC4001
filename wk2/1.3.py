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
    # if not (a < b and fa * fb < 0):
    #     raise ValueError('invalid input parameters or function')
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
    print(n)
    return n



def function(x):
    return x**2



print(bisect_root(function,-0.1,0.1,0.0001))

# c = (a + b) / 2
# fc = f(c)
# if fc == 0:
#     return c, n