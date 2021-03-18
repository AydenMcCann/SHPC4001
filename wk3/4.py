from scipy.integrate import solve_ivp
from pylab import *
import matplotlib.pyplot as plt


# Lotko-Volterra Model
def lotka_volterra(t, z, a, b, c, d):
    x, y = z
    return [a * x - b * x * y, -c * y + d * x * y]


# inputting model int solve_ivp
sol = solve_ivp(lotka_volterra, [0, 50], [1, 1], args=(2 / 3, 4 / 3, 1, 1),
                dense_output=True)

# assigning results to array
t = np.linspace(0, 50, 300)
z = sol.sol(t)

# plotting result
plt.plot(t, z.T)
plt.xlabel('t')
plt.ylabel('population')
plt.legend(['prey', 'predator'])
plt.title('Lotka-Volterra System')
plt.show()

# Phase portrait
a, b, c, d = 2 / 3, 4 / 3, 1, 1
xvalues, yvalues = np.meshgrid(np.arange(0, 6, 0.5), np.arange(0, 4, 0.5))
xdot = a * xvalues - b * xvalues * yvalues
ydot = -c * yvalues + d * xvalues * yvalues
plt.xlabel('Prey')
plt.ylabel('Predators')
plt.plot(1, 0.5, 'ro')
streamplot(xvalues, yvalues, xdot, ydot)
grid()
show()
