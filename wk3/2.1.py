import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

deltat = 0.5
tmax = 25
steps = int(tmax/deltat)
yvals = np.empty(51)

tau = 5
yvals[0] = 100  # initial condition N(0)
for i in range(steps): # Euler's Method
    yvals[i+1] = yvals[i]+deltat*(-1/tau)*yvals[i]

# plotting #
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
xvals = np.linspace(0,1,len(yvals))
plt.scatter(xvals, yvals, s=2, label="Euler's Method")
plt.plot(xvals, yvals)
plt.xlabel('t')  # axis labels
plt.ylabel('N(t)')
plt.legend(loc="upper right")

plt.show()  # show plot

