import numpy as np
from cython.parallel import parallel
import time
from libc.stdlib cimport rand, RAND_MAX #must not me import!
from cython import cdivision


tic = time.perf_counter() # begin timer


cdef:
    int iter = 10   # number of iterations
    n = 2           #dimensions
    points = 1000   # number of n-dimensional points to generate
    double mean
    double M2
    double r
    int numthreads = 12 # number of threads for random number generation
    vol = np.empty(iter)    # empty arrays/lists
    results = np.empty([3])
    point = []
    coords = np.empty([iter,n * points])
    newcoords = np.empty([iter,n,int(points)])


@cdivision(True) # random number generator (thanks Edric)
cdef double rand_double(double low, double high) nogil:
    cdef double r
    r = <double>rand()/<double>RAND_MAX
    r = (1 - r)*low + r*high
    return r


cdef MonteCarlo_NSphere_Vol(n, points):
    """ n = dimensions
        points = randomly generated points
    """
    counts = 0
    for count_loops in range(len(points[0,:])):
        r = np.linalg.norm(points[:,count_loops]) # calculates norm of n-dim coordinates
        if r < 1.0:
            counts = counts + 1
    return 2 ** n * (counts / len(points[0,:])) # volume estimate

""" Random numbers """
for i in range(iter):
    point = []
    while len(point) < n * points:  # creates num_threads number of random numbers at a time.
                                    # stops when this exceeds the desired number of points
        with nogil, parallel(num_threads=numthreads): # parallel generation of random numbers
            r = rand_double(-1, 1)
            with gil:
                point.append(r) # appends random numbers to list

    for m in range(n * points):
        coords[i,m] = point[m]  # converts to 2D-numpy array
    newcoords[i,:,:] = coords[i,:].reshape(n, int(points)) # reshapes into n-dim coordinate pairs
    vol[i] = MonteCarlo_NSphere_Vol(n, newcoords[i,:,:])    # calls the monte-carlo function
    if i == iter - 1:   # if final iteration is complete calculate final results
        mean = np.sum(vol) / iter
        M2 = (vol[-1] - mean) ** 2
        results[:] = (mean,M2/10,M2/(10-1))






toc = time.perf_counter()   # end timer


""" Printing results"""
print("Mean Volume ={}".format(results[0]))
print("Variance ={}".format(results[1]))
print("Completed in {} seconds".format(np.round(toc-tic,3)))


