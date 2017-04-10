import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

dt = 0.01
epsilon = dt**3
TF = 50000

x = np.zeros(TF)
y = np.zeros(TF)
x_temp = np.zeros(TF)
y_temp = np.zeros(TF)
ts = np.zeros(TF)
q = np.zeros(TF)


def x_dot_f(y_var):
    return np.sin(5*(np.sin(y_var)));

def y_dot_f(x_var):
    return np.cos(x_var);

#for t in range(TF - 1):
#    x_dot = x_dot_f(y[t])
#    y_dot = y_dot_f(x[t])
#
#    x[t+1] = x[t] + x_dot*dt
#    y[t+1] = y[t] + y_dot*dt
#
#e = plt.figure()
#plt.plot(x, y, 'ro')
#plt.title('Euler time stepping')
#e.show()

for t in range(TF - 1):
    if t == 0:
        x[t+1] = x[t] + x_dot_f(y[t])*dt
        y[t+1] = y[t] + y_dot_f(x[t])*dt
    elif t == 1:
        x[t+1] = x[t] + dt*(1.5*x_dot_f(y[t]) - 0.5*x_dot_f(y[t-1]))
        y[t+1] = y[t] + dt*(1.5*y_dot_f(x[t]) - 0.5*y_dot_f(x[t-1]))
    elif t == 2:
        x[t+1] = x[t] + dt*((23./12.)*x_dot_f(y[t]) - (4./3.)*x_dot_f(y[t-1]) + (5./12.)*x_dot_f(y[t-2]))
        y[t+1] = y[t] + dt*((23./12.)*y_dot_f(x[t]) - (4./3.)*y_dot_f(x[t-1]) + (5./12.)*y_dot_f(x[t-2]))
    else:
        x_temp[t+1] = x[t] + (dt/24.)*(55.*x_dot_f(y[t]) - 59.*x_dot_f(y[t-1]) + 37.*x_dot_f(y[t-2]) - 9.*x_dot_f(y[t-3]))
        y_temp[t+1] = y[t] + (dt/24.)*(55.*y_dot_f(x[t]) - 59.*y_dot_f(x[t-1]) + 37.*y_dot_f(x[t-2]) - 9.*y_dot_f(x[t-3]))

        x[t+1] = x[t] + (dt/24.)*(9.*x_dot_f(y_temp[t+1]) + 19.*x_dot_f(y[t]) - 5.*x_dot_f(y[t-1]) + x_dot_f(y[t-2]))
        y[t+1] = y[t] + (dt/24.)*(9.*y_dot_f(x_temp[t+1]) + 19.*y_dot_f(x[t]) - 5.*y_dot_f(x[t-1]) + y_dot_f(x[t-2]))


ab = plt.figure()
plt.plot(x, y, 'ro')
plt.title('AB/AM time stepping')
ab.show()

for t in range(TF - 1):
    if t == 0:
        x[t+1] = x[t] + x_dot_f(y[t])*dt
        y[t+1] = y[t] + y_dot_f(x[t])*dt
        ts[t+1] = dt
    elif t == 1:
        x[t+1] = x[t] + 1.5*dt*x_dot_f(y[t]) - 0.5*dt*x_dot_f(y[t-1])
        y[t+1] = y[t] + 1.5*dt*y_dot_f(x[t]) - 0.5*dt*y_dot_f(x[t-1])
        ts[t+1] = dt
    elif t == 2:
        x[t+1] = x[t] + dt*((23./12.)*x_dot_f(y[t]) - (4./3.)*x_dot_f(y[t-1]) + (5./12.)*x_dot_f(y[t-2]))
        y[t+1] = y[t] + dt*((23./12.)*y_dot_f(x[t]) - (4./3.)*y_dot_f(x[t-1]) + (5./12.)*y_dot_f(x[t-2]))
        ts[t+1] = dt
    elif t == 3:
        x_temp = x[t] + (dt/24.)*(55.*x_dot_f(y[t]) - 59.*x_dot_f(y[t-1]) + 37.*x_dot_f(y[t-2]) - 9.*x_dot_f(y[t-3]))
        y_temp = y[t] + (dt/24.)*(55.*y_dot_f(x[t]) - 59.*y_dot_f(x[t-1]) + 37.*y_dot_f(x[t-2]) - 9.*y_dot_f(x[t-3]))

        x[t+1] = x[t] + (dt/24.)*(9.*x_dot_f(y_temp) + 19.*x_dot_f(y[t]) - 5.*x_dot_f(y[t-1]) + x_dot_f(y[t-2]))
        y[t+1] = y[t] + (dt/24.)*(9.*y_dot_f(x_temp) + 19.*y_dot_f(x[t]) - 5.*y_dot_f(x[t-1]) + y_dot_f(x[t-2]))

        q_y = ((270./19.)*((epsilon*dt)/(np.abs(y[t+1] - y_temp))))**(0.25)
        q_y = np.floor(100.*q_y)/100.

        q_x = ((270./19.)*((epsilon*dt)/(np.abs(x[t+1] - x_temp))))**(0.25)
        q_x = np.floor(100.*q_x)/100.

        q[t+1] = min(q_x, q_y)

        ts[t+1] = dt
    else:
        x_temp = x[t] + q[t]*(dt/24.)*(55.*x_dot_f(y[t]) - 59.*x_dot_f(y[t-1]) + 37.*x_dot_f(y[t-2]) - 9.*x_dot_f(y[t-3]))
        y_temp = y[t] + q[t]*(dt/24.)*(55.*y_dot_f(x[t]) - 59.*y_dot_f(x[t-1]) + 37.*y_dot_f(x[t-2]) - 9.*y_dot_f(x[t-3]))

        x[t+1] = x[t] + q[t]*(dt/24.)*(9.*x_dot_f(y_temp) + 19.*x_dot_f(y[t]) - 5.*x_dot_f(y[t-1]) + x_dot_f(y[t-2]))
        y[t+1] = y[t] + q[t]*(dt/24.)*(9.*y_dot_f(x_temp) + 19.*y_dot_f(x[t]) - 5.*y_dot_f(x[t-1]) + y_dot_f(x[t-2]))

        ts[t+1] = q[t]*dt

        q_y = ((270./19.)*((epsilon*dt)/(np.abs(y[t+1] - y_temp))))**(0.25)
        q_y = np.floor(100.*q_y)/100.

        q_x = ((270./19.)*((epsilon*dt)/(np.abs(x[t+1] - x_temp))))**(0.25)
        q_x = np.floor(100.*q_x)/100.

        q[t+1] = min(q_x, q_y)

#
#v = plt.figure()
#plt.plot(x, y, 'ro')
#plt.title('Variable time stepping')
#v.show()
#
v2 = plt.figure()
plt.scatter(x[(ts < 0.0075)*(ts > 0.001)], y[(ts < 0.0075)*(ts > 0.001)], c = ts[(ts < 0.0075)*(ts > 0.001)], cmap = 'gnuplot', lw = 0, alpha = 0.5)
plt.colorbar()
plt.title('Variable time stepping')
v2.show()

tsp = plt.figure()
plt.plot(ts)
plt.plot([0, TF], [dt, dt])
tsp.show()

qp = plt.figure()
plt.plot(q)
qp.show()
