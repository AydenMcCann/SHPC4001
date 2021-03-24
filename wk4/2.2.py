from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

x = 5

cx = lambda t: np.cos((np.pi*t**2)/2) # defining cx and sx
sx = lambda t: np.sin((np.pi*t**2)/2)

steps = 10000 # number of steps to iterate through
dx = 10/steps

cxval = np.empty(steps)  # creating empty arrays for results
sxval = np.empty(steps)
errcx = np.empty(steps)
errsx = np.empty(steps)



for i in range(steps):
    cxval[i], errcx[i] = integrate.quad(cx, 0, -5+dx*i) # obtaining results from integrate.quad
    sxval[i], errsx[i] = integrate.quad(sx, 0, -5+dx*i)


# # plotting
# mpl.rcParams['font.family'] = 'Serif'
# plt.rcParams['font.size'] = 18
# plt.rcParams['axes.linewidth'] = 2
# plt.plot(np.linspace(-5,5,(len(sxval))), sxval, label = '$S_x$')
# plt.plot(np.linspace(-5,5,(len(sxval))), cxval, label = '$C_x$')
# plt.xlabel('x')  # axis labels
# plt.ylabel('')
# plt.legend(loc="upper right")
# plt.show()  # show plot


# parametric plot
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.plot(cxval, sxval)
plt.xlabel('$C_x$')  # axis labels
plt.ylabel('$S_x$')
plt.show()  # show plot
