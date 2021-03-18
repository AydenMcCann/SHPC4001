import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

tmax = 25
deltat = 0.5
steps = round(tmax/deltat)
yvals = np.empty(steps+2)
tau = 5

yvals[0] = 100 # initial condition N(0)
yvals[1] = yvals[0] + deltat * (-1 / tau) * yvals[0] #Euler's Method to establish N(1)
for i in range(steps): # Leap-Frog Method
    yvals[i+2] = yvals[i]+2*deltat*(-1/tau)*yvals[i+1]

# plotting #

mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
xvals = np.linspace(0,1,len(yvals))
plt.scatter(xvals, yvals, s=2, label="Leap-Frog Method")
plt.plot(xvals, yvals)
plt.xlabel('t')  # axis labels
plt.ylabel('N(t)')
plt.legend(loc="upper right")
plt.show()  # show plot

