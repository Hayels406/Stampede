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
TF = 200.0
dt = 0.01
NP = 1
b = 0.2
p = 3.
c = 0.5
A = 2*10**3
B = 0.08

pi = math.pi
t = 0.0
itime = 0

timeStepMethod = 'Euler'
updateMethod = 'Acceleration'
Walls = 'On'
###########################################################
###Variables
###########################################################
timesteps = int(round(TF/dt))*2
dog = np.zeros((timesteps+1, 2))
sheep = np.zeros((timesteps+1, 2))
dog[0] = np.array([2., 2.])#np.random.rand(2)*3.0+1.
sheep[0] = np.array([6. ,4.])#np.random.rand(2)*3.0 + 3.
epsilon = dt**6
sheepSize = 0.5
wallLeft = -20
wallRight = 20
wallBottom = -20
wallTop = 20
walls = [[0,1,-wallTop],[0,1,-wallBottom],[1,0,-wallLeft],[1,0,-wallRight]]#[[1, 0, -wallRight], [0, 1, -wallTop], [1, 0, -wallLeft], [0, 1, -wallBottom]]

bterm = np.zeros((timesteps+1,2))
aterm = np.zeros((timesteps+1))
normStuff = np.zeros((timesteps+1))
sheepVel = np.zeros((timesteps+1,2))
dogVel = np.zeros((timesteps+1,2))
sheepAccTerm2 = np.zeros((timesteps+1,2))
sheepAcc = np.zeros((timesteps+1,2))
dogAcc = np.zeros((timesteps+1,2))
Gfunc = np.zeros((timesteps+1,2))
Hfunc = np.zeros((timesteps+1,2))
dist_iw = np.zeros((timesteps+1,NP))
forceWalls = np.zeros((timesteps+1,2))
q = []

sheepVel[0][1] = 1
sheepVel[0][0] = 0.01
###########################################################
###Simulation
###########################################################
while t < TF:
    print t

    if updateMethod == 'Velocity':
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
        else:
            sys.exit('Invalid time step method 1')

    elif updateMethod == 'Acceleration':
        #Acceleration of sheep
        preyMinusPred = sheep[itime] - dog[itime]
        normStuff2 = np.transpose(np.tile(np.linalg.norm(preyMinusPred),(2,1)))
        Gfunc[itime] = normStuff2**(-1)
        sheepAccTerm2[itime] = Gfunc[itime]*preyMinusPred*(normStuff2**(-1))
        sheepAcc[itime] = -sheepVel[itime] + sheepAccTerm2[itime]
        print 'acc', sheepAcc[itime]

        #Acceleration of dog
        predMinusPrey = -(dog[itime] - sheep[itime])
        normStuff3 = np.transpose(np.tile(np.linalg.norm(predMinusPrey),(2,1)))
        Hfunc[itime] = 1./(normStuff3**1 + 1.)
        dogAcc[itime] = -dogVel[itime] + predMinusPrey*(normStuff3**(-1))*Hfunc[itime]


        if Walls == 'On':
            for w in range(len(walls)):
                [x,y,z] = walls[w]
                wall_point = [0.,0.]
                wall_point[0] = (x*z - sheep[itime,1]*x*y + sheep[itime,0]*y**2)/(x**2 + y**2)
                wall_point[1] = -(x/y)*wall_point[0] - z/y
                #wall_point = [(y*(y*sheep[itime, 0] - x*sheep[itime, 1]) + x*z)/(x**2 + y**2), (-z*(y*sheep[itime, 0] - x*sheep[itime, 1]) - y*z)/(x**2 + y**2)]
                dist_iw[itime] = abs(x*sheep[itime,0] + y*sheep[itime,1] + z)/np.sqrt(x**2 + y**2)#np.sqrt(((sheep[itime] - wall_point)**2).sum(axis = 0))
                print 'dist', dist_iw[itime]
                n_iw = (sheep[itime] - wall_point)/dist_iw[itime]
                print 'n', n_iw
                forceWalls[itime] = forceWalls[itime] + A*np.exp((sheepSize - dist_iw[itime])/B)*n_iw
                print 'force', forceWalls[itime]

            sheepAcc[itime] = forceWalls[itime] #sheepAcc[itime] +
            print 'acc', sheepAcc[itime]
        if timeStepMethod == 'Euler':
            #Euler time step
            dogVel[itime + 1] = dogVel[itime] + dogAcc[itime]*dt
            sheepVel[itime + 1] = sheepVel[itime] + sheepAcc[itime]*dt

            dog[itime + 1] = dog[itime] + dogVel[itime]*dt
            sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt
        elif timeStepMethod == 'Adaptive':
            #Adaptive time step
            if itime == 0:
                dogVel[itime + 1] = dogVel[itime] + dogAcc[itime]*dt
                sheepVel[itime + 1] = sheepVel[itime] + sheepAcc[itime]*dt
                sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt
                dog[itime + 1] = dog[itime] + dogVel[itime]*dt
            elif itime == 1:
                sheepVel[itime + 1] = sheepVel[itime] + 1.5*dt*sheepAcc[itime] - 0.5*dt*sheepAcc[itime - 1]
                dogVel[itime +1] = dogVel[itime] + dogAcc[itime]*dt
                sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt
                dog[itime + 1] = dog[itime] + dogVel[itime]*dt
            elif itime == 2:
                sheepVel[itime + 1] = sheepVel[itime] + dt*((23./12.)*sheepAcc[itime] - (4./3.)*sheepAcc[itime - 1] + (5./12.)*sheepAcc[itime - 2])
                dogVel[itime + 1] = dogVel[itime] + dogAcc[itime]*dt
                sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt
                dog[itime + 1] = dog[itime] + dogVel[itime]*dt
            elif itime == 3:
                sheepVelTemp = sheepVel[itime] + (dt/24.)*(55.*sheepAcc[itime] - 59.*sheepAcc[itime - 1] + 37.*sheepAcc[itime - 2] - 9.*sheepAcc[itime - 3])
                sheepAccTemp = (sheepVelTemp - sheepVel[itime])/dt
                sheepVel[itime + 1] = sheepVel[itime] + (dt/24.)*(9.*sheepAccTemp + 19.*sheepAcc[itime] - 5.*sheepAcc[itime - 1] + sheepAcc[itime - 2])
                dogVel[itime + 1] = dogVel[itime] + dogAcc[itime]*dt
                sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt
                dog[itime + 1] = dog[itime] + dogVel[itime]*dt
                q_f = np.min(((270./19.)*((epsilon*dt)/(np.abs(sheepVel[itime + 1] - sheepVelTemp))))**(0.25))
                q.append(np.floor(100.*q_f)/100.)
            else:
                sheepVelTemp = sheepVel[itime] + q[-1]*(dt/24.)*(55.*sheepAcc[itime] - 59.*sheepAcc[itime - 1] + 37.*sheepAcc[itime - 2] - 9.*sheepAcc[itime - 3])
                sheepAccTemp = (sheepVelTemp - sheepVel[itime])/(dt*q[-1])
                sheepVel[itime + 1] = sheepVel[itime] + q[-1]*(dt/24.)*(9.*sheepAccTemp + 19.*sheepAcc[itime] - 5.*sheepAcc[itime - 1] + sheepAcc[itime - 2])
                dogVel[itime + 1] = dogVel[itime] + dogAcc[itime]*dt*q[-1]
                sheep[itime + 1] = sheep[itime] + sheepVel[itime]*dt*q[-1]
                dog[itime + 1] = dog[itime] + dogVel[itime]*dt*q[-1]
                q_f = np.min(((270./19.)*((epsilon*dt)/(np.abs(sheepVel[itime + 1] - sheepVelTemp))))**(0.25))
                q.append(np.floor(100.*q_f)/100.)
        else:
            sys.exit('Invalid time step method 2')

    else:
        sys.exit('Invalid update method')



    ###########################################################
    ###Plotting
    ###########################################################
    if t == 0.0:
        plt.figure()
        dogTheta = np.zeros(1)
        sheepTheta = np.zeros(1)
        dogQuiver = plt.quiver(dog[itime, 0], dog[itime, 1], np.cos(dogTheta), np.sin(dogTheta), scale = 30, color = 'red')
        sheepQuiver = plt.quiver(sheep[itime, 0], sheep[itime, 1], np.cos(sheepTheta), np.sin(sheepTheta), scale = 30)
        plt.axis([wallLeft,wallRight,wallBottom,wallTop])
        plt.axes().set_aspect('equal')
        plotid = 0
        if Walls == 'On':
            plt.axhline(wallTop, color = 'r', lw = 5)
            plt.axhline(wallBottom, color = 'r', lw = 5)
            plt.axvline(wallLeft, color = 'r', lw = 5)
            plt.axvline(wallRight, color = 'r', lw = 5)

    if math.floor(t/dt) % 5 == 0:
        plotid = plotid + 1
        dogTheta = np.arctan2(dog[itime,1] - dog[itime - 1,1], dog[itime,0] - dog[itime - 1,0])
        sheepTheta = np.arctan2(sheep[itime,1] - sheep[itime - 1,1], sheep[itime,0] - sheep[itime - 1,0])

        dogQuiver.set_offsets(np.transpose([dog[itime, 0], dog[itime, 1]]))
        dogQuiver.set_UVC(np.cos(dogTheta),np.sin(dogTheta))
        sheepQuiver.set_offsets(np.transpose([sheep[itime, 0], sheep[itime, 1]]))
        sheepQuiver.set_UVC(np.cos(sheepTheta), np.sin(sheepTheta))
    plt.pause(0.005)

    if timeStepMethod == 'Adaptive':
        if itime <= 3:
            t += dt
            itime += 1
        else:
            t += dt*q[-1]
            itime += 1
    elif timeStepMethod == 'Euler':
        t += dt
        itime += 1
