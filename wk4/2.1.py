from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special as sc

cx = lambda t: np.cos((np.pi * t ** 2) / 2)  # defining cx and sx
sx = lambda t: np.sin((np.pi * t ** 2) / 2)

steps = 10000  # number of steps to iterate through
dx = 10 / steps

cxval = np.empty(steps)  # creating empty arrays for results
sxval = np.empty(steps)
errcx = np.empty(steps)
errsx = np.empty(steps)
S = np.empty(steps)
C = np.empty(steps)
Serr = np.empty(steps)
Cerr = np.empty(steps)

for i in range(steps):
    cxval[i], errcx[i] = integrate.quad(cx, 0, -5 + dx * i)  # obtaining results from integrate.quad
    sxval[i], errsx[i] = integrate.quad(sx, 0, -5 + dx * i)
    S[i], C[i] = sc.fresnel(-5 + dx * i)  # obtaining results from sc.fresnel
    Cerr[i] = np.abs(cxval[i] - C[i])  # calculating the absolute error
    Serr[i] = np.abs(sxval[i] - S[i])

# plotting
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.plot(np.linspace(-5, 5, (len(Cerr))), Cerr, label='$S_x$')
plt.plot(np.linspace(-5, 5, (len(Serr))), Serr, label='$C_x$')
plt.xlabel('x')  # axis labels
plt.ylabel('Error')
plt.legend(loc="upper right")
plt.show()  # show plot

