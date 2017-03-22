import numpy as np
from scipy.stats import itemfreq
import math
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import pylab as pylab
import sys
from sklearn.neighbors import KDTree
from scipy.spatial import Voronoi, voronoi_plot_2d
from pylab import savefig

plt.close()
###########################################################
###Parameters
###########################################################

TF = 500.                       # Final time
TSCREEN = 10                  # Screen update interval
dt = 0.1                    # Time step
NP = 400                     # Number of sheep
np.random.seed()

## Aux parameters
pi = math.pi
NX = 1.             # Resolution in x
NY = NX             # Resolution in y
t = 0.0             # Time starts at 0
show = False        # Show plots to screen
itime = 0

## Simulation parameters
a = float(sys.argv[2])
b = 0.2
c = float(sys.argv[1])
p = 3.

if c == 0.15:
    A = 0.00005
if c == 0.4:
    A = 0.01
if c == 0.8:
    A = 0.001
if c == 1.5:
    A == 0.01

massPred = 1.
massPrey = np.ones((NP,2))
velocityPrey = np.ones((NP,2))
velocityPred = np.array([0,0])

## Fuctions
def F(dist):
    return np.transpose(np.tile(1./dist - dist,(2,1)));

def G(dist):
    return np.transpose(np.tile(1./dist,(2,1)));

def H(dist):
    return np.transpose(np.tile(1/(dist**5 + 1),(2,1)));

def forcePrey(posnPrey, posnPred):
    return np.array(map(lambda j:(((posnPrey[j] - np.delete(posnPrey, j, 0))*np.transpose(np.tile(np.linalg.norm(posnPrey[j] - np.delete(posnPrey, j, 0), axis = 1)**(-1), (2,1)))*F(np.linalg.norm(posnPrey[j] - np.delete(posnPrey, j, 0), axis = 1))).sum(axis = 0)/NP + G(np.linalg.norm(posnPrey[j] - posnPred))*(posnPrey[j] - posnPred)/np.linalg.norm(posnPrey[j] - posnPred))[0].tolist(), range(NP)));


def forcePred(posnPrey, posnPred):
    return (1./NP)*((posnPrey - posnPred)/np.transpose(np.tile(np.linalg.norm(posnPrey - posnPred,axis=1) + A,(2,1)))*H(np.linalg.norm(posnPrey - posnPred,axis=1) + A)).sum(axis = 0);


np.random.seed(0)
positionPrey = np.random.rand(NP, 2)
posPrey = np.random.rand(NP)*2*pi
if c != 0.4:
    positionPrey[:,0] = (0.7+0.3*np.random.rand(NP))*np.cos(posPrey)
    positionPrey[:,1] = (0.7+0.3*np.random.rand(NP))*np.sin(posPrey)
       # Setting x and y for each prey np.array([[0.4, 0.6], [0.5, 0.8]])#
#positionPred = np.random.rand( 1, 2)          # Setting x and y for the predator

#positionPrey = np.array([[0, 0.2], [0.3, 0.1], [1., 0.8], [0.2, 0.7], [0.2, 0.1]])
positionPred = np.array([[5-5.0, 5-5.0]])

while (round(t, 3) < TF):
    print t
    #Update positionPrey
    #dpreydt = velocityPrey(positionPrey, positionPred)
    #dpreddt = velocityPred(positionPrey, positionPred)

    positionPreyOld = positionPrey
    positionPredOld = positionPred

    #Euler method
    accPred = forcePred(positionPrey, positionPred)/massPred
    accPrey = forcePrey(positionPrey, positionPred)/massPrey

    velocityPred = (velocityPred + accPred*dt)
    velocityPrey = (velocityPrey + accPrey*dt)

    positionPred = positionPred + velocityPred*dt
    positionPrey = positionPred + velocityPrey*dt

    meanPositionPrey = positionPrey.mean(axis = 0)
    # Ploting (Delete later)
    x = positionPrey[:, 0]
    y = positionPrey[:, 1]
    theta = np.arctan2(positionPrey[:, 1] - positionPreyOld[:, 1], positionPrey[:, 0] - positionPreyOld[:, 0])
    thetaPred = np.arctan2(positionPred[:, 1] - positionPredOld[:, 1], positionPred[:, 0] - positionPredOld[:, 0])
    xtheta = np.cos(theta)
    ytheta = np.sin(theta)
    if t == 0.0:
        plt.figure()
        q = plt.quiver(x, y, xtheta, ytheta, scale = 30)#, xtheta, ytheta, scale = 30, color = 'black')
        r = plt.quiver(positionPred[:,0], positionPred[:,1], np.cos(thetaPred), np.sin(thetaPred), color = 'red', scale = 30)
        #plt.plot(meanPositionPrey[0], meanPositionPrey[1], 'bo')
        #plt.xlim(NX/2. - 250, NX/2. + 250)
        #plt.ylim(NY/2. - 250, NY/2. + 250)
        plt.xlim(meanPositionPrey[0]-15., meanPositionPrey[0]+15.)
        plt.ylim(meanPositionPrey[1]-15., meanPositionPrey[1]+15.)
        plt.axes().set_aspect('equal')
    if itime%2 == 0:
        plt.xlim(meanPositionPrey[0]-15., meanPositionPrey[0]+15.)
        plt.ylim(meanPositionPrey[1]-15., meanPositionPrey[1]+15.)
        q.set_offsets(np.transpose([x, y]))
        q.set_UVC(xtheta,ytheta)
        r.set_offsets(np.transpose([positionPred[:,0], positionPred[:,1]]))
        r.set_UVC(np.cos(thetaPred), np.sin(thetaPred))
        #mean_x = x.mean()
        #mean_y = y.mean()
        #plt.xlim(mean_x - 20., mean_x + 20.)
        #plt.ylim(mean_y - 20., mean_y + 20.)
    plt.pause(0.005)

    t = t + dt
    itime = itime + 1
