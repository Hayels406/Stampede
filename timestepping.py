import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
epsilon = dt**4

x = np.zeros(300)
y = np.zeros(300)

def x_dot_f(y_var):
    return np.sin(y_var);

def y_dot_f(x_var):
    return np.cos(x_var);

for t in range(299):
    x_dot = x_dot_f(y[t])
    y_dot = y_dot_f(x[t])

    x[t+1] = x[t] + x_dot*dt
    y[t+1] = y[t] + y_dot*dt

e = plt.figure()
plt.plot(x, y, 'ro')
plt.title('Euler time stepping')
e.show()

for t in range(299):
    if t == 0:
        x[t+1] = x[t] + x_dot_f(y[t])*dt
        y[t+1] = y[t] + y_dot_f(x[t])*dt
    elif t == 1:
        x[t+1] = x[t] + 1.5*dt*x_dot_f(y[t]) - 0.5*dt*x_dot_f(y[t-1])
        y[t+1] = y[t] + 1.5*dt*y_dot_f(x[t]) - 0.5*dt*y_dot_f(x[t-1])
    elif t == 2:
        x[t+1] = x[t] + dt*((23./12.)*x_dot_f(y[t]) - (4./3.)*x_dot_f(y[t-1]) + (5./12.)*x_dot_f(y[t-2]))
        y[t+1] = y[t] + dt*((23./12.)*y_dot_f(x[t]) - (4./3.)*y_dot_f(x[t-1]) + (5./12.)*y_dot_f(x[t-2]))
    else:
        x_temp = x[t] + (dt/24.)*(55.*x_dot_f(y[t]) - 59.*x_dot_f(y[t-1]) + 37.*x_dot_f(y[t-2]) - 9.*x_dot_f(y[t-3]))
        y_temp = y[t] + (dt/24.)*(55.*y_dot_f(x[t]) - 59.*y_dot_f(x[t-1]) + 37.*y_dot_f(x[t-2]) - 9.*y_dot_f(x[t-3]))

        x[t+1] = x[t] + (dt/24.)*(9.*x_dot_f(y_temp) + 19.*x_dot_f(y[t]) - 5.*x_dot_f(y[t-1]) + x_dot_f(y[t-2]))
        y[t+1] = y[t] + (dt/24.)*(9.*y_dot_f(x_temp) + 19.*y_dot_f(x[t]) - 5.*y_dot_f(x[t-1]) + y_dot_f(x[t-2]))

        print ((270./19.)*(epsilon*dt/np.abs(y[t+1] - y_temp)))**(0.25)

ab = plt.figure()
plt.plot(x, y, 'ro')
plt.title('AB/AM time stepping')
ab.show()

for t in range(299):
    if t == 0:
        x[t+1] = x[t] + x_dot_f(y[t])*dt
        y[t+1] = y[t] + y_dot_f(x[t])*dt
    elif t == 1:
        x[t+1] = x[t] + 1.5*dt*x_dot_f(y[t]) - 0.5*dt*x_dot_f(y[t-1])
        y[t+1] = y[t] + 1.5*dt*y_dot_f(x[t]) - 0.5*dt*y_dot_f(x[t-1])
    elif t == 2:
        x[t+1] = x[t] + dt*((23./12.)*x_dot_f(y[t]) - (4./3.)*x_dot_f(y[t-1]) + (5./12.)*x_dot_f(y[t-2]))
        y[t+1] = y[t] + dt*((23./12.)*y_dot_f(x[t]) - (4./3.)*y_dot_f(x[t-1]) + (5./12.)*y_dot_f(x[t-2]))
    elif t == 3:
        x_temp = x[t] + (dt/24.)*(55.*x_dot_f(y[t]) - 59.*x_dot_f(y[t-1]) + 37.*x_dot_f(y[t-2]) - 9.*x_dot_f(y[t-3]))
        y_temp = y[t] + (dt/24.)*(55.*y_dot_f(x[t]) - 59.*y_dot_f(x[t-1]) + 37.*y_dot_f(x[t-2]) - 9.*y_dot_f(x[t-3]))

        x[t+1] = x[t] + (dt/24.)*(9.*x_dot_f(y_temp) + 19.*x_dot_f(y[t]) - 5.*x_dot_f(y[t-1]) + x_dot_f(y[t-2]))
        y[t+1] = y[t] + (dt/24.)*(9.*y_dot_f(x_temp) + 19.*y_dot_f(x[t]) - 5.*y_dot_f(x[t-1]) + y_dot_f(x[t-2]))

        q = round(((270./19.)*(epsilon*dt/np.abs(y[t+1] - y_temp)))**(0.25), 3)
    else:
        x_temp = x[t] + q*(dt/24.)*(55.*x_dot_f(y[t]) - 59.*x_dot_f(y[t-1]) + 37.*x_dot_f(y[t-2]) - 9.*x_dot_f(y[t-3]))
        y_temp = y[t] + q*(dt/24.)*(55.*y_dot_f(x[t]) - 59.*y_dot_f(x[t-1]) + 37.*y_dot_f(x[t-2]) - 9.*y_dot_f(x[t-3]))

        x[t+1] = x[t] + q*(dt/24.)*(9.*x_dot_f(y_temp) + 19.*x_dot_f(y[t]) - 5.*x_dot_f(y[t-1]) + x_dot_f(y[t-2]))
        y[t+1] = y[t] + q*(dt/24.)*(9.*y_dot_f(x_temp) + 19.*y_dot_f(x[t]) - 5.*y_dot_f(x[t-1]) + y_dot_f(x[t-2]))

        q = round(((270./19.)*(epsilon*dt/np.abs(y[t+1] - y_temp)))**(0.25), 3)



v = plt.figure()
plt.plot(x, y, 'ro')
plt.title('Variable time stepping')
v.show()
