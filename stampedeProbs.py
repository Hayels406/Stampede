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


###########################################################
###Parameters
###########################################################


#Parameters from command line

TF = 50.                     # Final time
iterations = 1                # Number of iterations of the simulation
TSCREEN = 10                  # Screen update interval
dt = 1.                    # Time step
SAVE = TF/dt                  # Interval to save to file
NP = 100                      # Number of sheep
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
    orientation = still*theta + theta_walking(tree, position, xxt, theta)*walking + theta_running(index_neighbours, xxp, xxt, position)*running
    return orientation;

def theta_walking(KDtree, positions_real_sheep, thetas_ghosts, theta_real_sheep):
    mean_theta = np.zeros(NP)
    idx = tree.query_radius(positions_real_sheep, r_0) # find neighbour ids for neighbours within radius
    for i in xrange(0,NP):
        dumidx = idx[i][idx[i] != (4*NP + i)] # remove identity particle
        if len(dumidx) > 0: # Some neighbours
            mean_theta[i] = np.arctan2(np.sin(thetas_ghosts[dumidx]).mean(axis = 0), np.cos(thetas_ghosts[dumidx]).mean(axis = 0))
        else:               # No neighbours
            mean_theta[i] = theta_real_sheep[i]
    phi = 2*np.random.rand(NP)*eta*pi - eta*pi
    mean_theta = mean_theta + phi
    return mean_theta;

def theta_running(idx_all, positions_ghosts, thetas_ghosts, positions_real_sheep):
    def f(dist, r_equ):
        f_r = min(1.0, (dist - r_equ)/r_equ)
        return f_r;

    def distance(index):
        dist = np.sqrt(np.sum(((position[index] - xxp[index_neighbours[index]])**2),1))
        return dist;


    def betaF(dist, r_equ):
        betaF1 = np.array(map(lambda x:beta*f(x, r_equ), dist))
        betaF2 = betaF1.reshape((len(betaF1), 1))
        return betaF2;

    betaGuy = map(lambda x:betaF(distance(x),r_e)*(np.divide(xxp[index_neighbours[x]] - position[x],np.transpose(np.tile(distance(x), 2).reshape((2,len(distance(x))))))), range(NP))
    trg = map(lambda x:map(list, zip(*[np.cos(xxt[x]), np.sin(xxt[x])]))*(xxb[x] == 2.).reshape((5,1)), index_neighbours)
    X,Y = map(list, zip(*map(lambda x,y:((x+y).mean(axis = 0)).tolist(), trg, betaGuy)))
    mean_theta = np.arctan2(Y,X)
    return mean_theta;

def change_behaviour(behavioural_vector):
    new_behav = np.zeros(NP)
    for i in range(len(behavioural_vector)):
        b = behavioural_vector[i]
        if b == 0.:
            #print i, b, ([(1-(p01(tree, position[i], i) + p012(position[i], index_neighbours[i], i))), p01(tree, position[i], i), p012(position[i], index_neighbours[i], i)])
            new_behav[i] = behaviour[i] #np.random.choice(np.arange(0,3), p =[(1-(p01(tree, position[i], i) + p012(position[i], index_neighbours[i], i))), p01(tree, position[i], i), p012(position[i], index_neighbours[i], i)])
        elif b == 1.:
            #print i, b, ([p10(tree, position[i], i), (1-(p10(tree, position[i], i) + p012(position[i], index_neighbours[i], i))), p012(position[i], index_neighbours[i], i)])
            new_behav[i] = behaviour[i] #np.random.choice(np.arange(0,3), p =[p10(tree, position[i], i), (1-(p10(tree, position[i], i) + p012(position[i], index_neighbours[i], i))), p012(position[i], index_neighbours[i], i)])
        else:
            #print i, b, ([p20(index_neighbours[i], i), 0, (1-p20(index_neighbours[i], i))])
            new_behav[i] = behaviour[i] #np.random.choice(np.arange(0,3), p =[p20(index_neighbours[i], i), 0, 1-p20(index_neighbours[i], i)])
            #if new_behav[i] == 0:
            #    stopped[i::NP] == 1
    return new_behav

def p01(KDTree, position_real_sheep, index):
    neighbours = tree.query_radius(position_real_sheep, r_0)[0]
    neighbours = neighbours[neighbours != (4*NP + index)]
    n_1 = sum(xxb[neighbours] == 1)
    prob01 = (1. + alpha*n_1)/(tau01)
    prob01 = 1. - np.exp(-1.*prob01*dt)
    return prob01

def p10(KDTree, position_real_sheep, index):
    neighbours = tree.query_radius(position_real_sheep, r_0)[0]
    neighbours = neighbours[neighbours != (4*NP + index)]

    n_0 = sum(xxb[neighbours] == 0)
    prob10 = (1. + alpha*n_0)/(tau10)

    prob10 = 1. - np.exp(-1.*prob10*dt)
    return prob10

def p012(position_real_sheep, idx, index):
    m_R = sum((xxb[idx]==2))
    mean_dist = np.sqrt((xxp[index_neighbours[index]][:,0] - np.tile(position[index][0],len(index_neighbours[index])))**2 + (xxp[index_neighbours[index]][:,1] - np.tile(position[index][1],len(index_neighbours[index])))**2).mean()
    prob012 = (1./tau012)*(mean_dist*(1. + alpha*m_R)/d_R)**delta
    prob012 = 1. - np.exp(-1.*prob012*dt)
    return prob012

def p20(idx, index):
    m_S = sum(stopped[idx])
    mean_dist = np.sqrt((xxp[index_neighbours[index]][:,0] - np.tile(position[index][0],len(index_neighbours[index])))**2 + (xxp[index_neighbours[index]][:,1] - np.tile(position[index][1],len(index_neighbours[index])))**2).mean()
    prob20 = (1./tau20)*((1.+ alpha*m_S)*d_S/mean_dist)**delta
    prob20 = 1. - np.exp(-1.*prob20*dt)
    return prob20

for l in xrange(iterations):
    plt.close()
    ## Sheep parameters
    v_1 = 0.15
    v_2 = 1.50
    r_0 = 1
    r_e = 1
    eta = 0.13
    beta = 0.8
    alpha = 13
    delta = 3
    d_R = 31.6
    d_S = 6.3
    tau01 = 35
    tau10 = 35
    tau012 = 100
    tau20 = 100

    stopped = np.zeros(NP*9)

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
    position = np.random.rand(NP, 2)*NX/10. + 4.5*NX/10.       # Setting x and y for each sheep
    theta = np.random.rand(NP)*2*pi           # Seting theta for each sheep
    behaviour = 2*np.ones(NP)#np.random.randint(0, 3, NP)   # Setting the behavioural states

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

        # Create Vor teselation
        vor = Voronoi(xxp)
        index_neighbours = map(lambda x:np.array([t[1] for t in [(b, a) for a, b in vor.ridge_dict.keys()] + vor.ridge_dict.keys() if t[0] == x]), range(4*NP, 5*NP)) #Calculates the Voronoi neighbours


        # Update angle
        theta = change_theta(behaviour)

        # Update behaviour
        #behaviour = change_behaviour(behaviour)


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
	    #savefig('/data/b1033128/Stampede/Animation/frame'+str(itime).zfill(6) +'.png') #plt.pause(0.05)
        plt.pause(0.05)

        t = t + dt;             # updating time
        itime = itime + 1;      # updating the iteger time (number of time steps done)

    # Re initialise parameters
    t = 0.0
    itime = 0
    pcounter = 0




print('Done')
