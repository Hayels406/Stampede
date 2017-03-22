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

TF = 200.                       # Final time
TSCREEN = 10                  # Screen update interval
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
tau = 0.5
B = np.ones(NP)*0.08
A = np.ones(NP)*2*10**3
k = 1.2*10**5
kappa = 2.4*10**5



###########################################################
###Functions
###########################################################
def g(x):
    if x < 0:
        return(0)
    else:
        return(x)

def f_ij(posi, posj, distij, ri, rj):
    r_ij = ri + rj
    if distij == 0.0:
        n_ij = np.array([0.0, 0.0])
    else:
        n_ij = (posi - posj)/distij
    t_ij = np.array([-n_ij[1], n_ij[0]])
    f = (A[0]*np.exp((r_ij - distij)/B[0]))*n_ij
    return(f)

def f_iw(posi, ri, w):
    #w in the form [a, b, c] where ax + by + c = 0 is the equation for the wall
    [ty,a,b,c] = w
    if ty == 's':
        dist_iw = abs(a*posi[0] + b*posi[1] + c)/np.sqrt(a**2 + b**2)
        wall_point = [(b*(b*posi[0] - a*posi[1]) - a*c)/(a**2 + b**2), (-a*(b*posi[0] - a*posi[1]) - b*c)/(a**2 + b**2)]
        n_iw = (posi - wall_point)/dist_iw
        f = np.array((A[0]*np.exp((ri - dist_iw)/B[0]))*n_iw)
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
        f = np.transpose(np.tile((A[0]*np.exp((ri - dist_iw)/B[0])), (2,1)))*n_iw
        f = f.mean(axis = 0)
    return(f)

def distance(pos):
    dist = np.array(map(lambda i:map(lambda j: np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2), range(NP)), range(NP)))
    return(dist)



###########################################################
###Simulation
###########################################################
plt.close()

position = np.array([[x,y] for x in np.arange(0,np.ceil(np.sqrt(NP))*1.,1.) for y in np.arange(0,np.ceil(np.sqrt(NP))*1.,1.)])#np.random.rand(NP, 2) + 7.5#np.array([[1., 5.], [6., 4.], [6., 3.5]])#
mass = np.ones(NP)*50.
v_0 = (np.ones(NP)*0.6).tolist()
e_0 = np.zeros((NP, 2))
e_0[:,1] = np.ones(NP)
theta = np.random.rand(NP)#*2*pi
v = np.transpose(np.array([np.cos(theta), np.sin(theta)]))
r = np.random.rand(NP)/10 + 0.25#np.array([0.2, 0.4, 0.3])#
t_matrix = np.zeros((NP, NP, 2))

w_int = str(sys.argv[1])
if w_int == 's':
    walls = [['s', 1, 0, -12], ['s', 0, 1, -12], ['s', 1, 0, 2], ['s', 0, 1, 2]]
elif w_int == 'c':
    walls = [['c',5,5,9]]
else:
    sys.exit('Not a valid wall')

while (round(t, 3) < TF):
    print t
    # Ploting (Delete later)
    x = position[:, 0]
    y = position[:, 1]
    theta = np.arctan2(v[:, 1], v[:, 0])
    xtheta = np.cos(theta)
    ytheta = np.sin(theta)
    #print np.arctan2(ytheta, xtheta)
    if t == 0.0:
        plt.figure()
        if w_int == 's':
            plt.plot([-2, 12], [-2, -2], lw = 4, color = 'red')
            plt.plot([-2, -2], [-2, 12], lw = 4, color = 'red')
            plt.plot([-2, 12], [12, 12], lw = 4, color = 'red')
            plt.plot([12, 12], [-2, 12], lw = 4, color = 'red')
        else:
            circle1=plt.Circle((walls[0][1],walls[0][2]),walls[0][3],color='r',fill = False, lw = 4)
            plt.gcf().gca().add_artist(circle1)
        q = plt.quiver(x, y, xtheta, ytheta, scale = 30)#, xtheta, ytheta, scale = 30, color = 'black')
        #plt.xlim(NX/2. - 250, NX/2. + 250)
        #plt.ylim(NY/2. - 250, NY/2. + 250)
        if w_int == 's':
            plt.xlim(-3., 13.)
            plt.ylim(-3., 13.)
        else:
            plt.xlim(-5., 15.)
            plt.ylim(-5., 15.)
        plt.axes().set_aspect('equal')
    if t > 0.0:
        q.set_offsets(np.transpose([x, y]))
        q.set_UVC(xtheta,ytheta)
        #mean_x = x.mean()
        #mean_y = y.mean()
        #plt.xlim(mean_x - 20., mean_x + 20.)
        #plt.ylim(mean_y - 20., mean_y + 20.)
    plt.pause(0.05)
    #if itime%TSCREEN == (TSCREEN - 1):
    #    savefig('/share/nobackup/b1033128/Walls/Animation'+str(w_int).upper()+'/frame'+str(itime).zfill(6) +'.png')
    d = distance(position)
    #n = n_mat(position, d)
    #t_matrix[:,:,0] = -n[:,:,1]
    #t_matrix[:,:,1] =  n[:,:,0]

    intrinsic_force = np.transpose(np.tile(mass, (2,1)))*np.array(map(lambda x:(v_0[x]*e_0[x] - v[x])/tau, range(NP)))
    extrinsic_force = map(lambda i:np.delete(map(lambda j:f_ij(position[i], position[j], d[i,j], r[i], r[j]), range(NP)), i, 0).sum(axis = 0),range(NP))
    wall_force = map(lambda i:np.array(map(lambda w:f_iw(position[i], r[i], w), walls)).sum(axis = 0), range(NP))
    force = intrinsic_force + extrinsic_force + wall_force
    acceleration = force/np.transpose(np.tile(mass, (2,1)))
    v = v + acceleration*dt

    position = position + v*dt

    if w_int == 's':
        print intrinsic_force[0]
        print extrinsic_force[0]
        print wall_force[0]

    t = t + dt
    itime = itime + 1
