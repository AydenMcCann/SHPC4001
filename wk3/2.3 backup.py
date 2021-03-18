import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

tmax = 25
deltat = 0.1
steps = round(tmax/deltat)
yvals = np.empty(steps+2)
tau = 5
yvals[0] = 100
yvals[1] = yvals[0] + deltat * (-1 / tau) * yvals[0]
for i in range(steps):
    yvals[i+2] = yvals[i]+2*deltat*(-1/tau)*yvals[i+1]


yvals2 = np.empty(steps+1)
yvals2[0] = 100
for i in range(steps):
    yvals2[i+1] = yvals2[i]+deltat*(-1/tau)*yvals2[i]


exact = np.empty(steps+2)
for i in range(steps):
    exact[i+1] = yvals2[i]+deltat*(-1/tau)*yvals2[i]







# plotting
xvals = np.linspace(0,1,len(yvals))
xvals2 = np.linspace(0,1,len(yvals2))
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.scatter(xvals, yvals, s=2, label="")

plt.scatter(xvals2, yvals2, s=2, label="")

plt.xlabel('t')  # axis labels
plt.ylabel('N(t)')
#plt.legend(loc="lower left")
plt.show()  # show plot

