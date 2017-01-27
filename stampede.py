import numpy as np
from scipy.stats import itemfreq
import math
import matplotlib.pyplot as plt
import os
from matplotlib import rc
import pylab as pylab
import sys
from sklearn.neighbors import KDTree
from scipy.spatial import Voronoi, voronoi_plot_2d


###########################################################
###Parameters
###########################################################


#Parameters from command line


TF = 0.200                    # Final time
iterations = 1                # Number of iterations of the simulation
TSCREEN = 10                  # Screen update interval
dt = 0.001                    # Time step
SAVE = TF/dt                  # Interval to save to file
NP = 100                      # Number of particles
np.random.seed()

# Setting up arrays to save data in
gorder = np.zeros([iterations,(int(TF/(TSCREEN*dt)))])
xxp = np.zeros([9*NP, 2])
xxt = np.zeros(9*NP)
xxb = np.zeros(9*NP)

# # Setting up directory to save to
# if runOn == 'server':
#     direct = '/data/b1033128/Stampede/'
#     if not os.path.exists(direct):
#         os.makedirs(direct)
# elif runOn == 'mac':
#     direct = 'temp/Stampede'
#     if not os.path.exists(direct):
#         os.makedirs(direct)
# elif runOn == 'topsy':
#     direct = './Stampede/'
#     if not os.path.exists(direct):
#         os.makedirs(direct)
# else:
#     direct = ''

# if not os.path.exists(direct+str(a)+'urms'):
#    os.makedirs(direct+str(a)+'urms')

def v(behavioural_vector):
    still = (behaviour == 0)
    walking = (behaviour == 1)
    running = (behaviour == 2)
    speed = 0.0*still + v_1*walking + v_2*running
    return speed;

def change_theta(behavioural_vector):
    still = (behaviour == 0)
    walking = (behaviour == 1)
    running = (behaviour == 2)
    orientation = still*theta + theta_walking(tree, position, xxt)*walking + 0.0*running
    return orientation;

def theta_walking(kdtree, positions_real_sheep, thetas_ghosts):
    mean_theta = np.zeros(NP)
    idx = tree.query_radius(positions_real_sheep, r_0) # find neighbour ids for neighbours within radius
    for i in xrange(0,NP):
        dumidx = idx[i][idx[i] != i] # remove identity particle
        if len(dumidx) > 0: # Some neighbours
            mean_theta[i] = np.arctan2(np.sin(thetas_ghosts[dumidx]).mean(axis = 0), np.cos(thetas_ghosts[dumidx]).mean(axis = 0))
        else:               # No neighbours
            mean_theta[i] = theta[i]
    phi = 2*np.random.rand(NP)*eta*pi - eta*pi
    mean_theta = mean_theta + phi
    return mean_theta;

def theta_running():
    mean_theta = np.zeros(NP)
    vor = Voronoi(position)
    idx = map(lambda x:np.array([t[1] for t in [(b, a) for a, b in vor.ridge_dict.keys()] + vor.ridge_dict.keys() if t[0] == x]), range(vor.npoints)) #Calculates the Voronoi neighbours

for l in xrange(iterations):
    plt.close()
    ## Sheep parameters
    v_1 = 45
    v_2 = 150
    r_0 = 1
    eta = 0.13

    ## Aux parameters
    pi = math.pi
    NX = 50             # Resolution in x
    NY = NX             # Resolution in y
    t = 0.0             # Time starts at 0
    show = False        # Show plots to screen
    itime = 0           # Number of time steps passed
    pcounter = 0


    ###########################################################
    ###Simulation
    ###########################################################
    #if itime == 0:
        #print backgroundNoise, 'a =', a, ',  m =', m,  ',  p =', p,  ',  l =', l
        #sys.stdout.flush()

    ## Initialise particles
    position = np.random.rand(NP, 2)*NX       # Setting x and y for each sheep
    theta = np.random.rand(NP)*2*pi             # Seting theta for each sheep
    behaviour = np.random.randint(0, 2, NP)# Setting the behavioural states

    print position
    print theta
    print behaviour

    np.random.seed(0)

    while (round(t, 3) < TF):

        ## Particles
        # Update position then angle

        # Speed in x and y
        speedx = v(behaviour)*np.cos(theta)
        speedy = v(behaviour)*np.sin(theta)

        # New location
        position[:,0] = np.mod(position[:,0] + dt*speedx, NX)
        position[:,1] = np.mod(position[:,1] + dt*speedy, NY)

        ## Flocking
        # Create a 9x9 grid of the fish to allow for full periodicity
        counter = 0
        for peri in xrange(-1,2):
            for perj in xrange(-1,2):
                xxp[counter*NP:(counter + 1)*NP, 0] = position[:, 0] + peri*NX
                xxp[counter*NP:(counter + 1)*NP, 1] = position[:, 1] + perj*NY
                xxt[counter*NP:(counter + 1)*NP] = theta[:].reshape(NP)
                xxb[counter*NP:(counter + 1)*NP] = behaviour[:].reshape(NP)
                counter = counter + 1

        # Create KD Tree
        tree = KDTree(xxp)

        theta = change_theta(behaviour)
#
        #if Fmodel == 1: # N nearest neighbours
            #dist, idx = tree.query(partx, neigh+2) # N neighbours and their distances
            #idx = idx[:, 1:neigh+1]
            #mean_theta = np.arctan2(np.sin(xxt[idx]).mean(axis=1), np.cos(xxt[idx]).mean(axis = 1))
            #closest[l, pcounter, :] = dist[0:NP,neigh]
            #furthest[l, pcounter, :] = dist[0:NP,neigh + 1]
#
        #elif Fmodel == 2: # Fixed radius
            #idx = tree.query_radius(partx, Rint) # find neighbour ids for neighbours within radius
            #mneigh = 0
            #mcounter = 0
            #for i in xrange(0,NP): #loops over each fish to calculate new direction and find the average number of neighbours
                #dumidx = idx[i][idx[i] != i] # remove identity particle
                #if len(dumidx) > 0: # Some neighbours
                    #mean_theta[i] = np.arctan2(np.sin(xxt[dumidx]).mean(axis = 0), np.cos(xxt[dumidx]).mean(axis = 0))
                    #mneigh = mneigh + len(dumidx)
                    #mcounter = mcounter + 1
                    #neigh_store[l, itime, i] = len(dumidx)
                #else:  # No neighbours
                    #mean_theta[i] = partt[i]
            #mneigh = mneigh/mcounter
#
        #partt = mean_theta
#
        ## Creating data
        #if itime%TSCREEN == (TSCREEN - 1):

        # Ploting (Delete later)
        x = position[:, 0]
        y = position[:, 1]
        xtheta = np.cos(np.squeeze(theta[:]))
        ytheta = np.sin(np.squeeze(theta[:]))
        if t == 0.0:
            plt.figure()
            q = plt.quiver(x, y, xtheta, ytheta, scale = 20)
            plt.xlim(0, NX)
            plt.ylim(0, NY)
            plt.axes().set_aspect('equal')
        if t > 0.0:
    	    q.set_offsets(np.transpose([x, y]))
    	    q.set_UVC(xtheta,ytheta)
	    plt.pause(0.05)

        t = t + dt;             # updating time
        itime = itime + 1;      # updating the iteger time (number of time steps done)

    # Re initialise parameters
    t = 0.0
    itime = 0
    pcounter = 0




print('Done')
