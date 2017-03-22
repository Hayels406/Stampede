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
TSCREEN = 50                  # Screen update interval
dt = 0.01                    # Time step
NP = 100                     # Number of sheep
np.random.seed()

## Aux parameters
pi = math.pi
NX = 1.             # Resolution in x
NY = NX             # Resolution in y
t = 0.0             # Time starts at 0
show = False        # Show plots to screen
itime = 0

## Simulation parameters
a = 0.08
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


tau = dt
Bw = np.ones(NP)*0.08
Aw = np.ones(NP)*2*10**3
k = 1.2*10**5
kappa = 2.4*10**5
predRadius = 0.25
w1 = 10
w2 = -10
massPrey = np.ones((NP,2))*50.
preyRadius = np.random.rand(NP)/10 + 0.25

###########################################################
###Functions
###########################################################

def velocityPreyIndex(index, postPrey, postPred):
    def innerSumFunc(index_j, index_k):
        return (postPrey[index_j] - postPrey[index_k])/(((postPrey[index_j] - postPrey[index_k])**2).sum()) - a*(postPrey[index_j] - postPrey[index_k])

    innersum = np.array(map(lambda x:innerSumFunc(index, x), np.delete(range(NP), index, 0))).sum(axis = 0)
    vel = (1./NP)*innersum + b*(postPrey[index] - postPred)/np.sqrt(((postPrey[index] - postPred)**2).sum())
    return vel.tolist()[0];

def velocityPrey(posPrey, posPred):
    #map(lambda j:((positionPrey[j] - np.delete(positionPrey, j, 0))*np.transpose(np.tile(np.linalg.norm(positionPrey[j] - np.delete(positionPrey, j, 0), axis = 1)**(-2) - a, (2,1))).sum(axis = 0))/NP + b*(positionPrey[j] - positionPred)/np.linalg.norm(positionPrey[j] - positionPred)**2, range(NP))
    return np.array(map(lambda j:(((posPrey[j] - np.delete(posPrey, j, 0))*np.transpose(np.tile(np.linalg.norm(posPrey[j] - np.delete(posPrey, j, 0), axis = 1)**(-2) - a, (2,1)))).sum(axis = 0)/NP + b*(posPrey[j] - posPred)/np.linalg.norm(posPrey[j] - posPred)**2)[0].tolist(), range(NP)));


def velocityPred(posPrey, posPred):
    vel = (c/NP)*((posPrey - posPred)/np.transpose(np.tile(np.linalg.norm(posPrey - posPred,axis=1)**p + A,(2,1)))).sum(axis = 0)
    return vel;

def f_iw(posi, ri, w):
    #w in the form [a, b, c] where ax + by + c = 0 is the equation for the wall
    [ty,a,b,c] = w
    if ty == 's':
        dist_iw = abs(a*posi[0] + b*posi[1] + c)/np.sqrt(a**2 + b**2)
        wall_point = [(b*(b*posi[0] - a*posi[1]) - a*c)/(a**2 + b**2), (-a*(b*posi[0] - a*posi[1]) - b*c)/(a**2 + b**2)]
        n_iw = (posi - wall_point)/dist_iw
        f = np.array((Aw[0]*np.exp((ri - dist_iw)/Bw[0]))*n_iw)
    elif ty == 'c':
        dist_iw = np.zeros(4)
        wall_point = np.zeros((4, 2))
        n_iw = np.zeros(4)
        PC = posi - np.array([a, b])
        dist_iw[0] = abs(np.sqrt(((PC)**2).sum()) - c)
        dist_iw[1] = 2*c - dist_iw[0]
        dist_iw[2] = np.sqrt(dist_iw[1]*(2*c - dist_iw[1]))
        dist_iw[3] = dist_iw[2]
        theta = np.arctan2(PC[1], PC[0])
        phi = theta - pi/2.
        wall_point[0] = [a + c*np.cos(theta), b + c*np.sin(theta)]
        wall_point[1] = [a - c*np.cos(theta), b - c*np.sin(theta)]
        wall_point[2] = [posi[0] - dist_iw[3]*np.cos(phi), posi[1] - dist_iw[3]*np.sin(phi)]
        wall_point[3] = [posi[0] + dist_iw[3]*np.cos(phi), posi[1] + dist_iw[3]*np.sin(phi)]
        n_iw = (np.tile(posi, (4,1)) - wall_point)/dist_iw.reshape(4,1)
        f = np.transpose(np.tile((Aw[0]*np.exp((ri - dist_iw)/Bw[0])), (2,1)))*n_iw
        f = f.mean(axis = 0)
    return(f)

def f_ij(posi, posj, distij, ri, rj):
    r_ij = ri + rj
    if distij == 0.0:
        n_ij = np.array([0.0, 0.0])
    else:
        n_ij = (posi - posj)/distij
    t_ij = np.array([-n_ij[1], n_ij[0]])
    f = (Aw[0]*np.exp((r_ij - distij)/Bw[0]))*n_ij
    return(f)

def distance(pos):
    dist = np.array(map(lambda i:map(lambda j: np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2), range(NP)), range(NP)))
    return(dist)

###########################################################
###Set Up
###########################################################
np.random.seed(0)
massPred = np.array([17])
#positionPrey = np.random.rand(NP, 2)
positionPrey = np.array([[x,y] for x in np.arange(0,np.ceil(np.sqrt(NP)),1.) for y in np.arange(0,np.ceil(np.sqrt(NP)),1.)])[0:NP]#np.random.rand(NP, 2) + 7.5#np.array([[1., 5.], [6., 4.], [6., 3.5]])#

#posPrey = np.random.rand(NP)*2*pi
#if c != 0.4:#
#    positionPrey[:,0] = (0.7+0.3*np.random.rand(NP))*np.cos(posPrey)
#    positionPrey[:,1] = (0.7+0.3*np.random.rand(NP))*np.sin(posPrey)
       # Setting x and y for each prey np.array([[0.4, 0.6], [0.5, 0.8]])#
#positionPred = np.random.rand( 1, 2)          # Setting x and y for the predator

#positionPrey = np.array([[0, 0.2], [0.3, 0.1], [1., 0.8], [0.2, 0.7], [0.2, 0.1]])
positionPred = np.array([[-5., -5.]])
walls = [['s', 1, 0, -w1], ['s', 0, 1, -w1], ['s', 1, 0, -w2], ['s', 0, 1, -w2]]#[['c',5,5,9]]#

vPredOld = np.zeros((int(TF/dt) + 1,2))
vPreyOld = np.zeros((int(TF/dt) + 1,NP,2))

###########################################################
###Simulation
###########################################################
while (round(t, 3) < TF):
    print t
    #Update positionPrey
    d = distance(positionPrey)
    positionPreyOld = positionPrey
    positionPredOld = positionPred


    #Euler method
    vPrey = velocityPrey(positionPrey, positionPred)
    vPred = velocityPred(positionPrey, positionPred)
    meanPositionPrey = positionPrey.mean(axis = 0)
    vPreyOld[itime+1] = vPrey
    vPredOld[itime+1] = vPred
    accPred = (vPred - vPredOld[itime])/tau
    accPrey = (vPrey - vPreyOld[itime])/tau

    forcePred = massPred*accPred + np.array(map(lambda w:f_iw(positionPred[0], predRadius, w), walls)).sum(axis = 0)
    forcePrey = massPrey*accPrey + map(lambda i:np.delete(map(lambda j:f_ij(positionPrey[i], positionPrey[j], d[i,j], preyRadius[i], preyRadius[j]), range(NP)), i, 0).sum(axis = 0),range(NP)) + map(lambda i:np.array(map(lambda w:f_iw(positionPrey[i], preyRadius[i], w), walls)).sum(axis = 0), range(NP))

    positionPred = positionPred + (vPred + (forcePred/massPred)*dt)*dt
    positionPrey = positionPrey + (vPrey + (forcePrey/massPrey)*dt)*dt

    # Ploting (Delete later)
    x = positionPrey[:, 0]
    y = positionPrey[:, 1]
    theta = np.arctan2(positionPrey[:, 1] - positionPreyOld[:, 1], positionPrey[:, 0] - positionPreyOld[:, 0])
    thetaPred = np.arctan2(positionPred[:, 1] - positionPredOld[:, 1], positionPred[:, 0] - positionPredOld[:, 0])
    xtheta = np.cos(theta)
    ytheta = np.sin(theta)
    if t == 0.0:
        plt.figure()
        plt.plot([w2, w1], [w2, w2], lw = 4, color = 'red')
        plt.plot([w2, w2], [w2, w1], lw = 4, color = 'red')
        plt.plot([w2, w1], [w1, w1], lw = 4, color = 'red')
        plt.plot([w1, w1], [w2, w1], lw = 4, color = 'red')
        plt.plot([w2 + 0.35, w1 - 0.35], [w2 + 0.35, w2 + 0.35], lw = 1, color = 'blue')
        plt.plot([w2 + 0.35, w2 + 0.35], [w2 + 0.35, w1 - 0.35], lw = 1, color = 'blue')
        plt.plot([w2 + 0.35, w1 - 0.35], [w1 - 0.35, w1 - 0.35], lw = 1, color = 'blue')
        plt.plot([w1 - 0.35, w1 - 0.35], [w2 + 0.35, w1 - 0.35], lw = 1, color = 'blue')
        q = plt.quiver(x, y, xtheta, ytheta, scale = 30)#, xtheta, ytheta, scale = 30, color = 'black')
        r = plt.quiver(positionPred[:,0], positionPred[:,1], np.cos(thetaPred), np.sin(thetaPred), color = 'red', scale = 30)
        #plt.plot(meanPositionPrey[0], meanPositionPrey[1], 'bo')
        #plt.xlim(NX/2. - 250, NX/2. + 250)
        #plt.ylim(NY/2. - 250, NY/2. + 250)
        plt.xlim(w2, w1)
        plt.ylim(w2, w1)
        plt.axes().set_aspect('equal')
    if itime%2 == 0:
        q.set_offsets(np.transpose([x, y]))
        q.set_UVC(xtheta,ytheta)
        r.set_offsets(np.transpose([positionPred[:,0], positionPred[:,1]]))
        r.set_UVC(np.cos(thetaPred), np.sin(thetaPred))
        #mean_x = x.mean()
        #mean_y = y.mean()
        #plt.xlim(mean_x - 20., mean_x + 20.)
        #plt.ylim(mean_y - 20., mean_y + 20.)
    plt.pause(0.005)
    #if itime%TSCREEN == (TSCREEN - 1):
    #    savefig('/share/nobackup/b1033128/Sheep/frame'+str(itime).zfill(6) +'.png')


    t = t + dt
    itime = itime + 1
