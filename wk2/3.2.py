from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

x_values = np.linspace(0, 314, 5500)
j_values = sp.jv(0, x_values)
j0_zero_guesses = x_values[:-1][j_values[:-1] * j_values[1:] < 0]
rootarray = np.empty(100)
error = np.empty(100)
docval = sp.jn_zeros(0, 100)


def f(x):
    return sp.jv(0, x)


def fd(x):
    return sp.jvp(0, x)


def sigfig(x, y):  # rounding to significant figures
    if x < 1:
        r = round(x, y)
        return r
    else:
        for i in range(len(str(x))):
            if 10 ** i < x <= 10 ** (i + 1):
                r = round(x, y - i - 1)
                return r


for i in range(len(rootarray)):  # calculating roots and rounding to 6 significant figures
    rootarray[i] = optimize.newton(f, j0_zero_guesses[i], fprime=fd, rtol=0.0001)
    rootarray[i] = sigfig(rootarray[i], 6)
    error[i] = np.abs(rootarray[i] - docval[i])

# visualize
fig, ax = plt.subplots()
ax.axhline(c='k', lw=1)
lines = ax.plot(x_values, j_values, '-b', j0_zero_guesses, np.zeros_like(j0_zero_guesses), 'or', rootarray,
                np.zeros_like(rootarray), 'og')
ax.set_xlabel('$x$')
ax.legend(lines, ('$J_0(x)$', '(approximate) zeros of $J_0(x)$', '(accurate) zeros of $J_0(x)$'))

plt.show()

print("Average error when compared to jn_zeros = {}".format(sum(error) / 100))
