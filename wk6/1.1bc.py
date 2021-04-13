import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# introducing parameters for the potential
v0 = 1.5
omega = 1
x0 = 35

# introducing parameters for the wave packet and nyquist condition
x1 = 10  # central position of wavepacket at t=0
k0 = 2
m = 2
eps = np.finfo(float).eps
sigma = 0.5

# animation constants
scale = 1  # vertically scales the wavefunction to allow meaningful superposition with potential
speed = 80  # causes the animation to skip frames; allows for faster animations without changing dt
frames = 60  # number of frames
fps = 30    # frames per second for the gif and matplotlib window animation

# introducing step sizes
dt = 0.01
dx = 0.1

# calculating max dx based on nyquist condition
dxmax = np.pi / (
        k0 + np.sqrt(2 * v0) + (1 / (np.sqrt(2) * sigma)) * np.sqrt(np.log((sigma / eps) * np.sqrt(2 / np.pi))))
if dx > dxmax:
    print("Maximum Δx for choice in other parameters = {}".format(dxmax))
    raise Exception("choice of dx in violation of Nyquist condition")

# create the x grid
x = np.arange(0, 100 + dx, dx)
N = len(x)

# create the initial wave packet
psi0 = np.exp(1j * k0 * x) * np.exp(-(x - x1) ** 2 / 4 * sigma ** 2) / ((2 * np.pi * sigma ** 2) ** (1 / 4))


# defining the potential
def v(x):
    """ defining the potential """
    return v0 / (np.cosh((x - x0) / omega) ** 2)

V = v(x)


# construct the 4th order FD matrix
g = -5j / (4 * m * dx ** 2) - 1j * V
a = 1j / (24 * m * dx ** 2)
diag = np.diag(g)
off_diag1 = np.diag([16 * a] * (N - 1), 1) + np.diag([16 * a] * (N - 1), -1)
off_diag2 = np.diag([-a] * (N - 2), 2) + np.diag([-a] * (N - 2), -2)
M = diag + off_diag1 + off_diag2

# create the time grid
t = np.arange(0, int(dt * speed * frames + 1), dt)  # upper bound ensures no unused timesteps are generated
steps = len(t)

# create an array containing wavefunctions for each step
y = np.zeros([steps, N], dtype=np.complex128)
y[0] = psi0

# the RK4 method
for i in range(0, steps - 1):
    k1 = np.dot(M, y[i])
    k2 = np.dot(M, y[i] + k1 * dt / 2)
    k3 = np.dot(M, y[i] + k2 * dt / 2)
    k4 = np.dot(M, y[i] + k3 * dt)
    y[i + 1] = y[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# plotting and animation
plt.style.use('seaborn-pastel')
fig = plt.figure()
ax = plt.axes(xlim=(0, 100), ylim=(0, 3))
line, = ax.plot([], [], lw=3)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    x = np.linspace(0, 100, int(100 / dx), dtype=int)
    yy = (np.abs(y[speed * i, int(1 / dx) * x]) ** 2) * scale
    line.set_data(x, yy)
    return line,


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=frames, interval=int((1/fps)*1000), blit=True, cache_frame_data=False)
plt.plot(x, (v(x)))
plt.xlabel('x')  # axis label
plt.show()  # show matplotlib window animation
anim.save('potential_barrier.gif',fps=fps)  # save animation to gif


