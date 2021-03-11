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
    print((a + b) / 2)
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
# plt.axvline(x=bisect_root(function, 0, 1, 0.00001, 500), color="red")
# plt.axhline(0, color='red')
# plt.show()  # show plot

steps = 10000 # number of iterations
xval = np.empty(steps)
xval[0] = 0
dt = 0.0001 # size of change in x per iteration
a0 = 0 # low estimate
b0 = 1 # high estimate
array = np.empty(steps)
convergence = np.empty(steps)

print((bisect_root(function, 0, 1, 0.000001, 500)))  # printing root at various accuracies
for i in range(steps):
    xval[i] = 1 - i * dt  # starting from 1 and reducing the minimum accuracy
    array[i] = (bisect_root(function, 0, 1, xval[i], 500))  # inputting values of the bisect root function into an array
    convergence[i] = (np.log(b0 - a0) - np.log(xval[i])) / (np.log(2))  # calculating analytically the convergence for
    # for the same values

# plotting parameters #
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.scatter(xval, array, s=2, label="Observed Numerical Convergence")
plt.scatter(xval, convergence, s=2, label="Analytical Convergence")
plt.xlabel('t (target precision)')  # axis labels
plt.ylabel('steps needed')
plt.legend(loc="upper right")

plt.show()  # show plot

# substituting to verify answers #
print(function(0.609375))
