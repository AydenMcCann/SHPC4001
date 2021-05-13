import numpy as np
import matplotlib.pyplot as plt
import time



tic = time.perf_counter() # begin timer



def MonteCarlo_NSphere_Vol(n, points):
    """ n = number of dimensions
        points = number of random points to generate in n-space"""
    counts = 0
    for count_loops in range(points):
        point = np.random.uniform(0, 1.0, n) # generates n random numbers from 0,1
        r = np.linalg.norm(point)   # calculates the norm of the n random numbers
        if r < 1.0:
            counts += 1
    return 2 ** n * (counts / points) # volume estimation


ndim = 2        # number of dimensions
iter = 2      # number of iterations to average over
points = 10000   # controls number of points
                # code performs iter number of iterations for number of points.
                # ie. 1,2,3.... pow points

vol = np.empty(iter)
results = np.empty([3, points])    # empty arrays

for k in range(points):
    for i in range(iter):
        vol[i] = MonteCarlo_NSphere_Vol(ndim, k+1)  # calling the volume function
        if i == iter - 1:   # if it has completed all iterations for a given
            mean = np.sum(vol) / iter # mean volume
            M2 = (vol[-1] - mean) ** 2  # square distance from mean
            results[:, k] = (mean,M2/iter,M2/(iter-1))  # mean, variance and sample variance


toc = time.perf_counter() # end timer



""" Plotting """
pi = np.empty(points)
xs = np.empty(points)
for k in range(points):
    xs[k] = k
    pi[k] = np.pi


# volume plot
plt.plot(xs,results[0])
plt.plot(xs,pi)
plt.ylabel("Mean Volume")
plt.xlabel("# Points")
plt.xscale("log")
plt.show()


# variance plot
variance = results[1]
plt.plot(xs[1:],variance[1:])
plt.ylabel("Variance")
plt.xlabel("# Points")
# plt.yscale("log")
plt.xscale("log")
plt.show()

print("Completed in {} seconds".format(np.round(toc-tic,3)))    # print timer