import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import special

deltax = np.empty(15)
steps = np.empty(15)
maxsteps = int(1 / 2 ** -15)

yvals = np.empty([maxsteps + 1, 15])
yvals[0] = 0  # initial value N(0)
k12 = np.empty([maxsteps, 15])
k2 = np.empty([maxsteps, 15])
exact = np.empty([maxsteps, 15])
error = np.empty([maxsteps, 15])
error2 = np.empty(15)

for k in range(15):  # loop over changing delta t values
    deltax[k] = 2 ** -k
    steps[k] = int(1 / deltax[k])
    for i in range(int(steps[k])):  # loop over the number of steps for each delta t
        k12[i, k] = yvals[i, k] + 0.5 * deltax[k] * (1 + 2 * (deltax[k] * i)) * yvals[i, k]
        k2[i, k] = 1 + 2 * (1 + deltax[k] / 2) * (k12[i, k])
        yvals[i + 1, k] = yvals[i, k] + k2[i, k] * deltax[k]
        exact[k] = (np.sqrt(np.pi) / 2) * np.exp((deltax[k] * i) ** 2) * special.erf(deltax[k] * i)
        error[i, k] = (np.abs(exact[i, k] - yvals[i + 1, k]))


error2 = error.sum(axis=0)  # sum the errors for each delta t

error3 = np.empty(15)


# gathering the error for t = 1
for i in range(15):
    error3[i] = error[2**i-1,i]
    print(2**i-1)
    print(i)
    print(".")


print(error3[14])



# plotting

deltats = np.empty(15)
for k in range(15):
    deltats[k] = 2 ** -k


mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.loglog(deltats, error3, label="Runge-Kutta Error")
plt.xlabel('Δt')  # axis labels
plt.ylabel('Error')
plt.legend(loc="lower right")
plt.show()  # show plot
