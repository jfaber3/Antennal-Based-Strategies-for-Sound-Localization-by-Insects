import random
import numpy as np
import matplotlib.pyplot as plt

def Hopf_dots(mu, w0, X, Y, Fx, Fy):
    X_dot = mu*X - w0*Y - X*(X**2 + Y**2) + Fx
    Y_dot = mu*Y + w0*X - Y*(X**2 + Y**2) + Fy
    return X_dot, Y_dot

def Hopf_RK2(mu, w0, X, Y, Fx, Fy, dt, D):
    X_noise = ((2*D*dt)**0.5)*random.gauss(0.0, 1.0)
    Y_noise = ((2*D*dt)**0.5)*random.gauss(0.0, 1.0)
    Xk1, Yk1 = Hopf_dots(mu, w0, X                   , Y                   , Fx, Fy)
    Xk2, Yk2 = Hopf_dots(mu, w0, X + Xk1*dt + X_noise, Y + Yk1*dt + Y_noise, Fx, Fy)
    new_X = X + (dt/2)*(Xk1 + Xk2) + X_noise
    new_Y = Y + (dt/2)*(Yk1 + Yk2) + Y_noise
    return new_X, new_Y

def Hopf_RK4(mu, w0, X, Y, Fx, Fy, dt):
    Xk1, Yk1 = Hopf_dots(mu, w0, X           , Y             , Fx, Fy)
    Xk2, Yk2 = Hopf_dots(mu, w0, X + Xk1*dt/2, Y + Yk1*dt/2  , Fx, Fy)
    Xk3, Yk3 = Hopf_dots(mu, w0, X + Xk2*dt/2, Y + Yk2*dt/2  , Fx, Fy)
    Xk4, Yk4 = Hopf_dots(mu, w0, X + Xk3*dt  , Y + Yk3*dt    , Fx, Fy)
    new_X = X + (dt/6)*(Xk1 + 2*Xk2 + 2*Xk3 + Xk4)
    new_Y = Y + (dt/6)*(Yk1 + 2*Yk2 + 2*Yk3 + Yk4)
    return new_X, new_Y

def HOPF(mu, w0, Forces_x, Forces_y, dt):
    if mu < 0:
        r0 = 0
    else:
        r0 = mu**0.5
    phi0 = 2*np.pi*random.random()
    N = len(Forces_x)
    X = np.zeros(N, dtype=float)
    Y = np.zeros(N, dtype=float)
    X[0] = r0*np.cos(phi0)
    Y[0] = r0*np.sin(phi0)
    for i in range(1, N):
        #X[i], Y[i] = Hopf_RK2(mu, w0, X[i-1], Y[i-1], Forces_x[i-1], Forces_y[i-1], dt, D)
        X[i], Y[i] = Hopf_RK4(mu, w0, X[i-1], Y[i-1], Forces_x[i-1], Forces_y[i-1], dt)
    return X, Y


tau_shift1 = 10
tau_shift2 = 3
tau_shift3 = 0.5
mu = 0.01
A0 = 0.01
dt = 0.001
num_cyc = 50
w0 = 1
N = int(round( num_cyc*2*np.pi/(w0*dt) ))
tt = np.linspace(-dt*N/2, dt*N/2, N)
phi0 = 2*np.pi*random.random()

Phi1 = np.pi/( 1 + np.e**(-tt/tau_shift1) )
Fx1, Fy1 = A0*np.cos(w0*tt + phi0 + Phi1), A0*np.sin(w0*tt + phi0 + Phi1)
x1, y1 = HOPF(mu, w0, Fx1, Fy1, dt)

Phi2 = np.pi/( 1 + np.e**(-tt/tau_shift2) )
Fx2, Fy2 = A0*np.cos(w0*tt + phi0 + Phi2), A0*np.sin(w0*tt + phi0 + Phi2)
x2, y2 = HOPF(mu, w0, Fx2, Fy2, dt)

Phi3 = np.pi/( 1 + np.e**(-tt/tau_shift3) )
Fx3, Fy3 = A0*np.cos(w0*tt + phi0 + Phi3), A0*np.sin(w0*tt + phi0 + Phi3)
x3, y3 = HOPF(mu, w0, Fx3, Fy3, dt)




fig = plt.figure(figsize=(10, 3))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.85, wspace=0.3, hspace=0.3)
ax1  = plt.subplot2grid((1, 3), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((1, 3), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((1, 3), (0,2), colspan=1, rowspan=1)
ax1.plot(tt, x1, color='black')
ax1.plot(tt, (Phi1/np.pi - 1/2), "--", color='red')
ax1.set_xlim(-100, 100)
ax1.set_ylim(-0.6, 0.6)
ax1.set_xlabel("Time")
ax1.set_ylabel(r'$x(t)$')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.plot(tt, x2, color='black')
ax2.plot(tt, (Phi2/np.pi - 1/2), "--", color='red')
ax2.set_xlim(-100, 100)
ax2.set_ylim(-0.6, 0.6)
ax2.set_xlabel("Time")
ax2.set_ylabel(r'$x(t)$')
ax2.set_xticks([])
ax2.set_yticks([])
ax3.plot(tt, x3, color='black')
ax3.plot(tt, (Phi3/np.pi - 1/2), "--", color='red')
ax3.set_xlim(-100, 100)
ax3.set_ylim(-0.6, 0.6)
ax3.set_xlabel("Time")
ax3.set_ylabel(r'$x(t)$')
ax3.set_xticks([])
ax3.set_yticks([])
ax1.text(-0.07, 1.15, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.07, 1.15, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.07, 1.15, "C", transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig2.jpeg', dpi=300)









'''
r = (x**2 + y**2)**0.5

plt.figure()
plt.plot(x)
plt.plot(Phi/np.pi - 1/2)
plt.plot(r, color='red')
plt.plot(Fx, color='black')

plt.figure()
plt.plot(x)
plt.plot(Phi/np.pi - 1/2)
'''


