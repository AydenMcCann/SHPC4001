import numpy as np  # for fast numerical computing with python
import matplotlib.pyplot as plt  # visualisation of the results

steps = 500  # time steps
tval = np.empty(steps + 1)  # time values
xval = np.empty(steps + 1)  # position (original) value at each time
vval = np.empty(steps + 1)  # velocity (original) value at each time
xvalcor = np.empty(steps + 1)  # position (corrected) value at each time
vvalcor = np.empty(steps + 1)  # velocity (corrected) value at each time

tfrac = np.empty(steps +1)
vfloor = np.empty(steps +1)

xval[0] = xvalcor[0] = 1.0  # initial height of the ball
vval[0] = vvalcor[0] = 0.0  # initial velocity of the ball

tval[0] = 0.0  # initial time (t=0 initially)
g = 9.8  # gravitational acceleration
dt = 0.1  # size of the time step

for i in range(steps):  # loop for 300 timesteps
    tval[i + 1] = tval[i] + dt
    xval[i + 1] = xval[i] + vval[i] * dt
    vval[i + 1] = vval[i] - g * dt
    vvalcor[i + 1] = vvalcor[i] - g * dt
    xvalcor[i + 1] = xvalcor[i] + ((vvalcor[i + 1] + vvalcor[i]) * dt) * 0.5
    xvalcor[1] = 1-(g*dt/2)*dt

    if xval[i] < 0:  # Reflect the motion of the ball when it strikes the surface
        vval[i + 1] = -vval[i]
        xval[i + 1] = 0

    if xvalcor[i+1] < 0:  # corrected reflection off surface

        tfrac[i+1] = xval[i]/(xval[i]+xvalcor[i+1]) # calculate which fraction of the timestep it will take to hit
        # exactly x=0

        vfloor[i] = np.sqrt(xvalcor[i]*2*g+(vvalcor[i])**2) # calculate the velocity it would be travelling at
        # exactly x=0
        vvalcor[i+1] = (vfloor[i] - g*dt*(1-tfrac[i+1]))  # calculate the velocity at the end of the timestep
        xvalcor[i+1] = ((vvalcor[i+1]+vfloor[i])/2)*(1-tfrac[i+1])*dt # calculate the position at the end of the
        # timestep
    else:
        tfrac[i] = 0
        vfloor[i] = 0

# save the results to file
f = open("bouncing_ball_results.csv", "w")  # create text file
f.write("xval, vval, vvalcor, xvalcor, tval\n")  # write column names
for i in range(steps + 1):  # write the results to file as text
    f.write("{0}, {1}, {2}, {3}, {4} \n".format(xval[i], vval[i], xvalcor[i], vvalcor[i], tval[i]))

f.close()  # close the file

x_datacor = xvalcor
x_data = xval
t_data = tval




plt.scatter(t_data, x_datacor)  # plot the results in a scatter plot
plt.scatter(t_data, x_data)
plt.xlabel('x')  # axis labels
plt.ylabel('t')
plt.show()  # show plot