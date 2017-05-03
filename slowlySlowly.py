import numpy as np
import math
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import pylab as pylab
import sys
from pylab import savefig

plt.close()

###########################################################
###Parameters
###########################################################
TF = 2.
dt = 0.1
NP = 1
b = 0.2
p = 3.
c = 0.5

pi = math.pi
t = 0.0
itime = 0

timeStepMethod = 'Adaptive'
###########################################################
###Variables
###########################################################
timesteps = int(round(TF/dt))
dog = np.zeros((timesteps+1, 2))
sheep = np.zeros((timesteps+1, 2))
dog[0] = np.random.rand(2)*3.0
sheep[0] = np.random.rand(2)*3.0 + 3.
epsilon = dt**4

bterm = np.zeros((timesteps+1,2))
aterm = np.zeros((timesteps+1))
normStuff = np.zeros((timesteps+1))
sheepVel = np.zeros((timesteps+1,2))
dogVel = np.zeros((timesteps+1,2))
q = []


###########################################################
###Simulation
###########################################################
while t < TF:
    print t

    #Velocity of sheep
    #distMatrixNotMe = np.array(map(lambda j:sheep[itime,j] - np.delete(sheep[itime], j, 0),range(NP)))
    #normStuff[itime] = np.transpose(np.tile(np.transpose(np.linalg.norm(distMatrixNotMe, axis = 2)**(-2)),(2,1,1)))
    preyMinusPred = sheep[itime] - dog[itime]
    normstuff2 = np.transpose(np.tile(np.linalg.norm(preyMinusPred),(2,1)))
    bterm[itime] = b*preyMinusPred*normstuff2**(-2)
    sheepVel[itime] = bterm[itime]#(distMatrixNotMe*(normStuff2 - a)).sum(axis = 1)/NP + bterm[itime]

    #Velocity of dog
    distDogSheep = sheep[itime] - dog[itime]
    frac = distDogSheep*np.transpose(np.tile((np.linalg.norm(distDogSheep))**(-p),(2,1)))
    dogVel[itime] = (c/NP)*frac.sum(axis=0)

    if timeStepMethod == 'Euler':
        #Euler time step
        dog[itime + 1] = dog[itime] + dogVel[itime]*dt
        sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt
    elif timeStepMethod == 'Adaptive':
        #Adaptive time step
        if itime == 0:
            sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt
            dog[itime + 1] = dog[itime] + dogVel[itime]*dt
        elif itime == 1:
            sheep[itime + 1] = sheep[itime] + 1.5*dt*sheepVel[itime] - 0.5*dt*sheepVel[itime - 1]
            dog[itime +1] = dog[itime] + dogVel[itime]*dt
        elif itime == 2:
            sheep[itime + 1] = sheep[itime] + dt*((23./12.)*sheepVel[itime] - (4./3.)*sheepVel[itime - 1] + (5./12.)*sheepVel[itime - 2])
            dog[itime + 1] = dog[itime] + dogVel[itime]*dt
        elif itime == 3:
            sheepTemp = sheep[itime] + (dt/24.)*(55.*sheepVel[itime] - 59.*sheepVel[itime - 1] + 37.*sheepVel[itime - 2] - 9.*sheepVel[itime - 3])
            sheepVelTemp = (sheepTemp - sheep[itime])/dt
            sheep[itime + 1] = sheep[itime] + (dt/24.)*(9.*sheepVelTemp + 19.*sheepVel[itime] - 5.*sheepVel[itime - 1] + sheepVel[itime - 2])
            dog[itime + 1] = dog[itime] + dogVel[itime]*dt
            q_f = np.min(((270./19.)*((epsilon*dt)/(np.abs(sheep[itime + 1] - sheepTemp))))**(0.25))
            q.append(np.floor(100.*q_f)/100.)
        else:
            sheepTemp = sheep[itime] + q[-1]*(dt/24.)*(55.*sheepVel[itime] - 59.*sheepVel[itime - 1] + 37.*sheepVel[itime - 2] - 9.*sheepVel[itime - 3])
            sheepVelTemp = (sheepTemp - sheep[itime])/(dt*q[-1])
            sheep[itime + 1] = sheep[itime] + q[-1]*(dt/24.)*(9.*sheepVelTemp + 19.*sheepVel[itime] - 5.*sheepVel[itime - 1] + sheepVel[itime - 2])
            dog[itime + 1] = dog[itime] + dogVel[itime]*dt*q[-1]
            q_f = np.min(((270./19.)*((epsilon*dt)/(np.abs(sheep[itime + 1] - sheepTemp))))**(0.25))
            q.append(np.floor(100.*q_f)/100.)

    ###########################################################
    ###Plotting
    ###########################################################
    if t == 0.0:
        plt.figure()
        dogTheta = np.zeros(1)
        sheepTheta = np.zeros(1)
        dogQuiver = plt.quiver(dog[itime, 0], dog[itime, 1], np.cos(dogTheta), np.sin(dogTheta), scale = 30, color = 'red')
        sheepQuiver = plt.quiver(sheep[itime, 0], sheep[itime, 1], np.cos(sheepTheta), np.sin(sheepTheta), scale = 30)
        plt.axis([-20,20,-20,20])
    else:
        dogTheta = np.arctan2(dog[itime,1] - dog[itime - 1,1], dog[itime,0] - dog[itime - 1,0])
        sheepTheta = np.arctan2(dog[itime,1] - dog[itime - 1,1], dog[itime,0] - dog[itime - 1,0])

        dogQuiver.set_offsets(np.transpose([dog[itime, 0], dog[itime, 1]]))
        dogQuiver.set_UVC(np.cos(dogTheta),np.sin(dogTheta))
        sheepQuiver.set_offsets(np.transpose([sheep[itime, 0], sheep[itime, 1]]))
        sheepQuiver.set_UVC(np.cos(sheepTheta), np.sin(sheepTheta))
    plt.pause(0.005)

    if itime <= 3:
        t += dt
        itime += 1
    else:
        t += dt*q[-1]
        itime += 1
