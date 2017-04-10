import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import csv
plt.close()

TF = 300.
dt = 1.
NP = 100
t = 0.
v = 0.1
neigh = 5
eta = 0.0
repulsion = 0.3
size = 600
plot = 1
itime = 0

position = np.random.rand(NP, 2)*size/2.
theta = np.random.rand(NP)*2*pi

joined = np.zeros((NP, 3))
joined[:,0:2] = position
joined[:,2] = theta

np.savetxt("vicsekData/data"+str(itime)+".csv", joined, delimiter=",")

while (t < TF):
    if plot == 1:
        x = position[:, 0]
        y = position[:, 1]
        xtheta = np.cos(np.squeeze(theta[:]))
        ytheta = np.sin(np.squeeze(theta[:]))
        if t == 0.:
            plt.figure()
            q = plt.quiver(x, y, xtheta, ytheta)
            plt.axes().set_aspect('equal')
            plt.xlim(-size/2., size)
            plt.ylim(-size/2., size)
        else:
            q.set_offsets(np.transpose([x, y]))
            q.set_UVC(xtheta, ytheta)
        plt.pause(0.005)

    # Create KD Tree
    tree = KDTree(position)
    dist, idx = tree.query(position, neigh+1) # N neighbours and their distances
    idx = idx[:, 1:neigh+1]
    mean_theta = np.arctan2(np.sin(theta[idx]).mean(axis=1), np.cos(theta[idx]).mean(axis = 1))
    phi = np.random.rand(NP)*pi - pi/2.
    for i in range(NP):
        if dist[i, 1] < repulsion:
            dumidx = idx[i][dist[i,1:] < repulsion]
            diff = -position[i] + position[idx[i,1]]
            mean_theta[i] = pi + np.arctan2(np.sin(theta[dumidx]).mean(axis=0), np.cos(theta[dumidx]).mean(axis = 0))

    theta = mean_theta + eta*phi
    position[:,0] = (position[:,0] + np.cos(theta)*v*dt)
    position[:,1] = (position[:,1] + np.sin(theta)*v*dt)

    joined = np.zeros((NP, 3))
    joined[:,0:2] = position
    joined[:,2] = theta
    itime  = itime + 1

    np.savetxt("vicsekData/data"+str(itime)+".csv", joined, delimiter=",")

    t = t + dt
