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

TF = 0.5                       # Final time
TSCREEN = 10                  # Screen update interval
dt = 0.01                    # Time step
NP = 400                      # Number of sheep
np.random.seed()

## Aux parameters
pi = math.pi
NX = 1.             # Resolution in x
NY = NX             # Resolution in y
t = 0.0             # Time starts at 0
show = False        # Show plots to screen
itime = 0

## Simulation parameters
a = 1.
b = 0.2
c = 0.4
p = 3.

## Fuctions
def velocityPrey(index):
    def innerSumFunc(index_j, index_k):
        #print (positionPrey[index_j] - positionPrey[index_k])/(((positionPrey[index_j] - positionPrey[index_k])**2).sum()) - a*(positionPrey[index_j] - positionPrey[index_k])
        return (positionPrey[index_j] - positionPrey[index_k])/(((positionPrey[index_j] - positionPrey[index_k])**2).sum()) - a*(positionPrey[index_j] - positionPrey[index_k])

    innersum = np.array(map(lambda x:innerSumFunc(index, x), np.delete(range(NP), index, 0))).sum(axis = 0)
    vel = (1./NP)*innersum + b*(positionPrey[index] - positionPred)/(((positionPrey[index] - positionPred)**2).sum())
    return vel.tolist()[0];

positionPrey = np.random.rand(NP, 2)          # Setting x and y for each prey
#positionPred = np.random.rand( 1, 2)          # Setting x and y for the predator

#positionPrey = np.array([[0, 0.2], [0.3, 0.1], [1., 0.8], [0.2, 0.7], [0.2, 0.1]])
positionPred = np.array([[0.5, 0.6]])

while (round(t, 3) < TF):
    print t
    #Update positionPrey
    dpreydt = np.array(map(lambda x:velocityPrey(x), range(NP)))
    dpreddt = (c/NP)*((positionPrey - positionPred)/(np.transpose(np.tile(((positionPrey - positionPred)**p).sum(axis = 1),2).reshape((2,NP))))).sum(axis = 0)

    positionPreyOld = positionPrey
    positionPredOld = positionPred

    positionPrey = positionPrey + dpreydt*dt
    positionPred = positionPred + dpreddt*dt

    # Ploting (Delete later)
    x = positionPrey[:, 0]
    y = positionPrey[:, 1]
    theta = np.arctan2(positionPreyOld[:, 1] - positionPrey[:, 1], positionPreyOld[:, 0] - positionPrey[:, 0]) + pi
    thetaPred = np.arctan2(positionPredOld[:, 1] - positionPred[:, 1], positionPredOld[:, 0] - positionPred[:, 0]) + pi
    xtheta = np.cos(theta)
    ytheta = np.sin(theta)
    #print np.arctan2(ytheta, xtheta)
    if t == 0.0:
        plt.figure()
        q = plt.quiver(x, y, xtheta, ytheta, scale = 30)#, xtheta, ytheta, scale = 30, color = 'black')
        r = plt.quiver(positionPred[:,0], positionPred[:,1], np.cos(thetaPred), np.sin(thetaPred), color = 'red', scale = 30)
        #plt.xlim(NX/2. - 250, NX/2. + 250)
        #plt.ylim(NY/2. - 250, NY/2. + 250)
        plt.xlim(-1.5, 2.5)
        plt.ylim(-1.5, 2.5)
        plt.axes().set_aspect('equal')
    if t > 0.0:
        q.set_offsets(np.transpose([x, y]))
        q.set_UVC(xtheta,ytheta)
        r.set_offsets(np.transpose([positionPred[:,0], positionPred[:,1]]))
        r.set_UVC(np.cos(thetaPred), np.sin(thetaPred))
        #mean_x = x.mean()
        #mean_y = y.mean()
        #plt.xlim(mean_x - 20., mean_x + 20.)
        #plt.ylim(mean_y - 20., mean_y + 20.)
    plt.pause(0.05)

    t = t + dt
    itime = itime + 1
