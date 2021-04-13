import numpy as np
import matplotlib.pyplot as plt

# introducing the parameters for the potential
v0 = 1.5
omega = 1
x0 = 35

# introducing the potential
def v(x):
    """ defining the potential """
    return v0 / (np.cosh((x - x0) / omega) ** 2)


# creating the x grid
xvals = np.linspace(0, 100, 500)

# plotting
plt.plot(xvals, v(xvals), label = "V(x)")
plt.legend(loc="upper right")
plt.show()


