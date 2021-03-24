import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def f(x):  # defining function
    return 3 + 2 * np.cos(3 * np.sqrt(x))


def midpoint(f, x1, x2, dx):
    """Midpoint rule implementation: f: function, x1,x2: lower and upper limits, dx: change in x size
        """
    n = round((x2 - x1) / dx)
    integralvals = np.empty(n)
    for k in range(n):
        x = x1 + k * dx
        integralvals[k] = (f((x + x + dx) / 2) * dx)
    finalvalue = integralvals.sum(axis=0)
    return finalvalue


def trapmeth(f, x1, x2, dx):
    """Trapezoidal rule implementation: f: function, x1,x2: lower and upper limits, dx: change in x size
        """
    n = round((x2 - x1) / dx)
    integralvals = np.empty(n)
    for k in range(n):
        x = x1 + k * dx
        integralvals[k] = (f(x) + f(x + dx)) * (dx / 2)
    finalvalue = integralvals.sum(axis=0)
    return finalvalue


def simpsons(f, x1, x2, dx):
    """Simpson's rule implementation: f: function, x1,x2: lower and upper limits, dx: change in x size
        """
    n = round((x2 - x1) / dx)
    integralvals = np.empty(n)
    for k in range(n):
        x = x1 + k * dx
        integralvals[k] = (f(x) + 4 * f(x + 0.5*dx) + f(x + dx)) * (dx / 6)
    finalvalue = integralvals.sum(axis=0)
    return finalvalue


def quadrature(f, x1, x2, dx):
    """Gaussian Quadrature rule implementation: f: function, x1,x2: lower and upper limits, dx: change in x size
        """
    n = round((x2 - x1) / dx)
    integralvals = np.empty(n)
    for k in range(n):
        x = x1 + k * dx
        integralvals[k] = 0.5*dx*(f(x+0.5*dx*(1-1/np.sqrt(3)))+f(x+0.5*dx*(1+1/np.sqrt(3))))
    finalvalue = integralvals.sum(axis=0)
    return finalvalue

lowpow = 15 # lowest power to reduce deltax to. For example 15 -> e-15
num = lowpow*2

traperr = np.empty(num)     # creating empty arrays
miderr = np.empty(num)
simperr = np.empty(num)
quaderr = np.empty(num)
xvals = np.empty(num)

fn = 3 + (4/9)*(-1+np.cos(3)+3*np.sin(3))   # Analytical Result of integral


for x in range(num):        # calculating absolute errors for a given deltax
    deltax = 2**(-x/2)
    xvals[x] = 2**(-x)
    print(num - x)

    traperr[x] = np.abs(trapmeth(f, 0, 1, deltax)-fn)
    miderr[x] = np.abs(midpoint(f, 0, 1, deltax)-fn)
    simperr[x] = np.abs(simpsons(f, 0, 1, deltax)-fn)
    quaderr[x] = np.abs(quadrature(f, 0, 1, deltax)-fn)

# plotting
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.loglog(xvals, traperr, label="Trapezoidal rule")
plt.plot(xvals, miderr, label="Midpoint rule")
plt.plot(xvals, simperr, label="Simpson’s rule")
plt.plot(xvals, quaderr, label="Gaussian quadrature")
plt.xlabel('Δt')  # axis labels
plt.ylabel('Error')
plt.legend(loc="upper left")
plt.show()  # show plot1.py
