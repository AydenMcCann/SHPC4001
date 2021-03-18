import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

num = 15
tmax = 25
steps = np.empty(num)
yvals = np.empty([round(tmax / (2 ** -num)), num]) # ensuring arrays are big enough by assigning them the max length
yvals2 = np.empty([round(tmax / (2 ** -num)), num])
exact = np.empty([round(tmax / (2 ** -num)), num])
for k in range(num):
    deltat = 2 ** -k
    steps[k] = tmax / deltat    # defining the number of steps for the given delta t
    tau = 5
    yvals[0, k] = 100   # defining N(0)
    yvals[1, k] = yvals[0, k] + deltat * (-1 / tau) * yvals[0, k]   # N(1) using forward euler method
    for i in range(int(steps[k])):
        yvals[i + 2, k] = yvals[i, k] + 2 * deltat * (-1 / tau) * yvals[i + 1, k]   # Leap-frog method
    yvals2[0, k] = 100
    for i in range(int(steps[k])):
        yvals2[i + 1, k] = yvals2[i, k] + deltat * (-1 / tau) * yvals2[i, k]    # Euler method
    exact[0, k] = 100
    for i in range(int(steps[k])):  # calculating exact analytical result
        exact[i,k] = 100*np.exp((-i*deltat)/tau)


erroreul = np.empty([round(tmax / (2 ** -num)), num])   # creating empty arrays for the error results
errorlea = np.empty([round(tmax / (2 ** -num)), num])
errorsumlea = np.empty(15)
errorsumeul = np.empty(15)

for i in range(round(tmax / (2 ** -num))):
    for k in range(num):
        errorlea[i,k] = np.abs(exact[i,k] - yvals[i,k])     # calculating errors
        erroreul[i,k] = np.abs(exact[i,k] - yvals2[i,k])

for k in range(num):
    deltat = 2 ** -k
    steps[k] = tmax / deltat
    errorlea[k] = errorlea[k]/steps[k]  # summing errors
    erroreul[k] = erroreul[k]/steps[k]
    errorsumlea =errorlea.sum(axis=0)
    errorsumeul =erroreul.sum(axis=0)



# plotting
deltats = np.empty(num)
for k in range(num):
    deltats[k] = 2 ** -k
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.scatter(deltats, errorsumeul, s=2, label="Average Error Euler")
plt.plot(deltats, errorsumeul)
plt.scatter(deltats, errorsumlea, s=2, label="Average Error Leapfrog")
plt.loglog(deltats, errorsumlea)
plt.xlabel('Δt')  # axis labels
plt.ylabel('Error')
plt.legend(loc="upper right")
plt.show()  # show plot
