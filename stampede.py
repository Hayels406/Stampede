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

TF = 6#2.*10**5                      # Final time
iterations = 1                # Number of iterations of the simulation
TSCREEN = 10                  # Screen update interval
dt = 1.                       # Time step
SAVE = TF/dt                  # Interval to save to file
NP = 5                        # Number of sheep
np.random.seed()

# Setting up arrays to save data in
gorder = np.zeros([iterations,(int(TF/(TSCREEN*dt)))])
xxp = np.zeros([9*NP, 2])
xxt = np.zeros(9*NP)
xxb = np.zeros(9*NP)

## Sheep parameters
v_1 = 0.15
v_2 = 1.50
r_0 = 10.0
r_e = 1.0
eta = 0.13
beta = 0.9#1.75
alpha = 13#5
delta = 3#2
d_R = 31.6
d_S = 6.3
tau01 = 35
tau10 = 35
tau012 = 100
tau20 = 100

## Aux parameters
pi = math.pi
NX = 100             # Resolution in x
NY = NX             # Resolution in y
t = 0.0             # Time starts at 0
show = False        # Show plots to screen

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

def v_vec(behavioural_vector):
    speed = 0.0*(behavioural_vector == 0) + v_1*(behavioural_vector == 1) + v_2*(behavioural_vector == 2)
    return speed;

def change_theta(behavioural_vector):
    orientation = (behavioural_vector == 0)*theta + theta_walking(position, xxt, theta)*(behavioural_vector == 1) + theta_running(index_neighbours, xxp, xxt, position)*(behavioural_vector == 2)
    return orientation;

def theta_walking(positions_real_sheep, thetas_ghosts, theta_real_sheep):
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
        if len(index_neighbours[index]) >0:
            dist = np.sqrt(np.sum(((position[index] - xxp[index_neighbours[index]])**2),1))
        else:
            dist = np.nan
        return dist;

    def unitVector(index):
        if np.isnan(distance(index)).any():
            e_ij = np.nan
        else:
            e_ij = (np.divide(xxp[idx_all[index].tolist()] - position[index],np.transpose(np.tile(distance(index), 2).reshape((2,len(distance(index)))))))
        return e_ij;

    def betaF(r_equ, index):
        if np.isnan(distance(index)).any():
            betaF3 = np.array([[np.nan, np.nan]])
        else:
            betaF1 = np.array(map(lambda x:beta*f(x, r_equ), distance(index)))
            betaF2 = betaF1.reshape((len(betaF1), 1))
            betaF3 = betaF2*unitVector(index)
        return betaF3;



    betaGuy = map(lambda x:betaF(r_e, x), range(NP))
    trg = map(lambda x:map(list, zip(*[np.cos(xxt[x.tolist()])*(xxb[x.tolist()] ==2), np.sin(xxt[x.tolist()])*(xxb[x.tolist()] == 2)])), idx_all)
    for i in range(NP):
        if len(trg[i]) == 0:
            trg[i] = [[np.nan, np.nan]]
    X,Y = map(list, zip(*map(lambda x,y:((x+y).sum(axis = 0)).tolist(), trg, betaGuy)))
    mean_theta = np.arctan2(Y,X)
    mean_theta[np.isnan(mean_theta)] = theta[np.isnan(mean_theta)]

    return mean_theta;

def change_behaviour(behavioural_vector):
    new_behav = np.zeros(NP)
    for i in range(len(behavioural_vector)):
        b = behavioural_vector[i]
        if b == 0.:
            b1 = np.random.choice(np.arange(0,2), p =[(1-p01(position[i], i)), p01(position[i], i)])
            b2 = np.random.choice(2*np.arange(0,2), p =[(1-p012(position[i], index_neighbours[i], i)), p012(position[i], index_neighbours[i], i)])
            if b1 == 1 and b2 == 2:
                new_behav[i] = np.random.choice(2*np.arange(0,2), p = [0.5, 0.5])
            elif b1 == 1:
                new_behav[i] = 1
            elif b2 == 2:
                new_behav[i] = 2
            else:
                new_behav[i] = 0
            if stopped[i] == 1:
                stopped[i::NP] = 0
        elif b == 1.:
            b0 = np.random.choice(np.arange(0,2), p =[p10(position[i], i), 1-p10(position[i], i)])
            b2 = np.random.choice(np.arange(1,3), p =[(1-p012(position[i], index_neighbours[i], i)), p012(position[i], index_neighbours[i], i)])
            if b0 == 0 and b2 == 2:
                new_behav[i] = np.random.choice(2*np.arange(0,2), p = [0.5, 0.5])
            elif b0 == 0:
                new_behav[i] = 0
            elif b2 == 2:
                new_behav[i] = 2
            else:
                new_behav[i] = 1
            if new_behav[i] == 0:
                stopped[i::NP] = 1
        else:
            new_behav[i] = np.random.choice(2*np.arange(0,2), p =[p20(index_neighbours[i], i), 1-p20(index_neighbours[i], i)])
            if new_behav[i] == 0:
                stopped[i::NP] = 1
        #if new_behav[i] == 2:
        #    new_behav[i] = behavioural_vector[i]
    return new_behav

def p01(position_real_sheep, index):
    neighbours = tree.query_radius(position_real_sheep.reshape((-1,2)), r_0)[0]
    neighbours = neighbours[neighbours != (4*NP + index)]
    n_1 = sum(xxb[neighbours] == 1)
    prob01 = (1. + alpha*n_1)/(tau01)
    prob01 = 1. - np.exp(-1.*prob01*dt)
    return prob01

def p10(position_real_sheep, index):
    neighbours = tree.query_radius(position_real_sheep.reshape((-1,2)), r_0)[0]
    neighbours = neighbours[neighbours != (4*NP + index)]

    n_0 = sum(xxb[neighbours] == 0)
    prob10 = (1. + alpha*n_0)/(tau10)

    prob10 = 1. - np.exp(-1.*prob10*dt)
    return prob10

def p012(position_real_sheep, idx, index):
    m_R = sum((xxb[idx.tolist()]==2))
    if len(idx) > 0:
        mean_dist = np.sqrt((xxp[index_neighbours[index]][:,0] - np.tile(position[index][0],len(index_neighbours[index])))**2 + (xxp[index_neighbours[index]][:,1] - np.tile(position[index][1],len(index_neighbours[index])))**2).mean()
        prob012 = (1./tau012)*(mean_dist*(1. + alpha*m_R)/d_R)**delta
        prob012 = 1. - np.exp(-1.*prob012*dt)
    else:
        mean_dist = np.nan
        prob012 = 1.0
    return prob012

def p20(idx, index):
    m_S = sum(stopped[idx.tolist()])
    if len(idx) > 0:
        mean_dist = np.sqrt((xxp[index_neighbours[index]][:,0] - np.tile(position[index][0],len(index_neighbours[index])))**2 + (xxp[index_neighbours[index]][:,1] - np.tile(position[index][1],len(index_neighbours[index])))**2).mean()
        prob20 = (1./tau20)*((1.+ alpha*m_S)*d_S/mean_dist)**delta
        prob20 = 1. - np.exp(-1.*prob20*dt)
    else:
        prob20 = 0.
    return prob20

for l in xrange(iterations):
    plt.close()

    stopped = np.zeros(NP*9)

    t = 0.
    itime = 0           # Number of time steps passed
    pcounter = 0


    ###########################################################
    ###Simulation
    ###########################################################
    #if itime == 0:
        #print backgroundNoise, 'a =', a, ',  m =', m,  ',  p =', p,  ',  l =', l
        #sys.stdout.flush()

    ## Initialise particles
    position = np.array([[ 54.,  46.],
       [ 54.,  54.],
       [ 50.,  48.],
       [ 47.,  51.],
       [ 45.,  45.]])#np.random.rand(NP, 2)*10. - 5. + NX/2.       # Setting x and y for each sheep
    theta = np.array([(2./3.)*pi, pi/6., pi/2., pi/2., pi/2.])#np.random.rand(NP)*2*pi           # Seting theta for each sheep
    behaviour = np.ones(NP)#np.random.randint(0, 3, NP)   # Setting the behavioural states

    np.random.seed(0)

    while (round(t, 3) < TF):
        ## Particles
        # Update position then angle
        print position
        # Speed in x and y
        speedx = v_vec(behaviour)*np.cos(theta)
        speedy = v_vec(behaviour)*np.sin(theta)

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

        print theta
        # Update angle
        theta = change_theta(behaviour)

        # Update behaviour
        print behaviour
        behaviour = change_behaviour(behaviour)


        # Ploting (Delete later)
        x = position[:, 0]
        y = position[:, 1]
        xtheta = np.cos(np.squeeze(theta[:]))
        ytheta = np.sin(np.squeeze(theta[:]))
        if t == 0.0:
            plt.figure()
            q = plt.quiver(x[behaviour == 0], y[behaviour == 0], xtheta[behaviour == 0], ytheta[behaviour == 0], scale = 30, color = 'black')
            r = plt.quiver(x[behaviour == 1], y[behaviour == 1], xtheta[behaviour == 1], ytheta[behaviour == 1], scale = 30, color = 'blue')
            s = plt.quiver(x[behaviour == 2], y[behaviour == 2], xtheta[behaviour == 2], ytheta[behaviour == 2], scale = 30, color = 'red')
            #plt.xlim(NX/2. - 250, NX/2. + 250)
            #plt.ylim(NY/2. - 250, NY/2. + 250)
            plt.xlim(0, NX)
            plt.ylim(0, NY)
            plt.axes().set_aspect('equal')
        if t > 0.0:
    	    q.set_offsets(np.transpose([x[behaviour == 0], y[behaviour == 0]]))
    	    q.set_UVC(xtheta[behaviour == 0],ytheta[behaviour == 0])
            r.set_offsets(np.transpose([x[behaviour == 1], y[behaviour == 1]]))
    	    r.set_UVC(xtheta[behaviour == 1],ytheta[behaviour == 1])
            s.set_offsets(np.transpose([x[behaviour == 2], y[behaviour == 2]]))
    	    s.set_UVC(xtheta[behaviour == 2],ytheta[behaviour == 2])
            #mean_x = x.mean()
            #mean_y = y.mean()
            #plt.xlim(mean_x - 20., mean_x + 20.)
            #plt.ylim(mean_y - 20., mean_y + 20.)
	#savefig('/data/b1033128/Stampede/Animation/frame'+str(itime).zfill(6) +'.png') #plt.pause(0.05)
        plt.pause(0.05)
        print t
        t = t + dt;             # updating time
        itime = itime + 1;      # updating the iteger time (number of time steps done)

    # Re initialise parameters
    t = 0.0
    itime = 0
    pcounter = 0




print('Done')
