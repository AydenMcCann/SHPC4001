import numpy as np  # for fast numerical computing with python
import matplotlib.pyplot as plt  # visualisation of the results
import matplotlib as mpl

steps = 2000 # time steps
tval = np.empty(steps + 1)  # time values
xval = np.empty(steps + 1)  # position (original) value at each time
vval = np.empty(steps + 1)  # velocity (original) value at each time
xvalcor = np.empty(steps + 1)  # position (corrected) value at each time
vvalcor = np.empty(steps + 1)  # velocity (corrected) value at each time

tfrac = np.empty(steps +1)
vfloor = np.empty(steps +1)

xval[0] = xvalcor[0] = 0.0  # initial height of the ball
vval[0] = vvalcor[0] = 5.0  # initial velocity of the ball

tval[0] = 0.0  # initial time (t=0 initially)
g = 9.8  # gravitational acceleration
dt = 0.001  # size of the time step

for i in range(steps):  # loop over timesteps
    tval[i + 1] = tval[i] + dt
    xval[i + 1] = xval[i] + vval[i] * dt
    vval[i + 1] = vval[i] - g * dt
    vvalcor[i + 1] = vvalcor[i] - g * dt
    xvalcor[i + 1] = xvalcor[i] + ((vvalcor[i + 1] + vvalcor[i]) * dt) * 0.5 #changed to use average velocity between current and last timestep

    if xval[i] < 0:  # Reflect the motion of the ball when it strikes the surface
        vval[i + 1] = -vval[i]*0.9
        xval[i + 1] = 0

    if xvalcor[i+1] < 0:  # corrected reflection off surface
        tfrac[i+1] = xval[i]/(xval[i]+xvalcor[i+1]) # calculate which fraction of the timestep it will take to hit
        # x=0
        vfloor[i] = np.sqrt(xvalcor[i]*2*g+(vvalcor[i])**2) # calculate the velocity it would be travelling at
        # x=0
        vvalcor[i+1] = 0.9*(vfloor[i] - g*dt*(1-tfrac[i+1]))  # calculate the velocity at the end of the timestep
        xvalcor[i+1] = ((vvalcor[i+1]+vfloor[i])/2)*(1-tfrac[i+1])*dt # calculate the position at the end of the
        # timestep
    else:
        tfrac[i] = 0
        vfloor[i] = 0

# save the results to file
f = open("bouncing_ball_results.csv", "w")  # create text file
f.write("xval, vval, vvalcor, xvalcor, tval\n")  # write column names
for i in range(steps + 1):  # write the results to file as text
    f.write("{0}, {1}, {2}, {3}, {4} \n".format(xval[i], vval[i], vvalcor[i], xvalcor[i], tval[i]))

f.close()  # close the file

x_datacor = xvalcor
x_data = xval
t_data = tval
v_data = abs(vval)
v_datacor = abs(vvalcor)

mpl.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.scatter(t_data, x_datacor, s =2, label = "Corrected")  # plot the results in a scatter plot
plt.scatter(t_data, x_data, s =2, label = "Original")
#plt.plot((0,steps*dt), (1,1), linestyle = '-', color='red')
plt.xlabel('time (s)')  # axis labels
plt.ylabel('position (m)')

plt.legend(loc="upper left")



plt.show()  # show plot