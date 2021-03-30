import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift


def psi(x, t):
    return np.exp(-(x ** 2) / (2 * (1 + 1j * t))) / (np.sqrt(np.sqrt(np.pi) * (1 + 1j * t)))


x_vals = np.linspace(-20, 20, 200)  # creating empty arrays for values
t_vals = np.array([0, 1, 5])
psi_vals = np.empty([3, 200], dtype=np.complex128)
psi_d = np.empty([3, 200])
psi_fft = np.empty([3, 200])

for i in range(3):  # iterating through 3 timesteps and 200 x positions to solve psi, its' fft etc
    for k in range(200):
        psi_vals[i, k] = psi(x_vals[k], t_vals[i])
        psi_d[i, k] = np.real(np.conj(psi_vals[i, k]) * (psi_vals[i, k]))
    psi_fft[i] = abs(fftshift(fft(psi_vals[i]))) ** 2
    psi_fft[i] = psi_fft[i] ** 2

# plotting #
exactxs = np.linspace(0, 4 * np.pi, 200)  # higher resolution to compare quality of transform
mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.xlabel('x')  # axis labels

""" Comment-out one of the two following sections to plot either Position or Momentum"""

# postition
for i in range(3):
    plt.plot(x_vals, psi_d[i, :], label='$|\psi(x,{})|^2$'.format(t_vals[i]))
plt.title('Position')
plt.xlabel('x')  # axis labels
plt.legend(loc="upper left")
plt.show()

# momentum
for i in range(3):
    plt.plot(x_vals, psi_fft[i,:],label='$FFT(\psi(x,{})$'.format(t_vals[i]))
plt.title('Momentum')
plt.xlabel('p')  # axis labels
plt.legend(loc="upper left")

plt.show()
