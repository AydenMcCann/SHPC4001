import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def f(x):  # defining function
    return np.sin(x)


def fd(x):  # first derivative of f(x)
    return np.cos(x)


def forward_d(f, x, h):  # defining the three methods
    df = (f(x + h) - f(x)) / h
    return df


def backward_d(f, x, h):
    df = (f(x) - f(x - h)) / h
    return df


def central_d(f, x, h):
    df = (f(x + h) - f(x - h)) / (2 * h)
    return df


forward_darray = np.empty([100, 15])  # creating empty arrays for results
backward_darray = np.empty([100, 15])
central_darray = np.empty([100, 15])

forward_error = np.empty([100, 15])
backward_error = np.empty([100, 15])
central_error = np.empty([100, 15])

forward_errav = np.empty(15)
backward_errav = np.empty(15)
central_errav = np.empty(15)

xvals = np.linspace(0, 2 * np.pi, num=100)  # creating x values to iterate over

for i in range(15):
    h = 10 ** (-i - 1)
    for j in range(100):
        forward_darray[j, i] = forward_d(f, xvals[j], h)
        backward_darray[j, i] = backward_d(f, xvals[j], h)
        central_darray[j, i] = central_d(f, xvals[j], h)

        forward_error[j, i] = np.abs(forward_darray[j, i] - fd(xvals[j]))
        backward_error[j, i] = np.abs(backward_darray[j, i] - fd(xvals[j]))
        central_error[j, i] = np.abs(central_darray[j, i] - fd(xvals[j]))

        forward_errav[i] = sum(forward_error[:, i]) / 100
        backward_errav[i] = sum(backward_error[:, i]) / 100
        central_errav[i] = sum(central_error[:, i]) / 100

# plotting #

mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
xrange = np.linspace(10 ** -15, 0.1, 15)
plt.scatter(xrange, central_errav, s=2, label="Central Error")
plt.loglog(xrange, central_errav)
plt.scatter(xrange, forward_errav, s=2, label="Forward Error")
plt.plot(xrange, forward_errav)
plt.scatter(xrange, backward_errav, s=2, label="Backward Error")
plt.plot(xrange, backward_errav)
plt.xlabel('h')  # axis labels
plt.ylabel('error')
plt.legend(loc="lower left")

plt.show()  # show plot
