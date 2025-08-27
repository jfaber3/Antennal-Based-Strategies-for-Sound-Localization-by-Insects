import numpy as np
import matplotlib.pyplot as plt
import code
from scipy.integrate import solve_ivp
from scipy import signal

clrs = ["#FF1F5B","#FFC61E", "#009ADE", "#00CD6C", "#AF58BA", "#F28522"]

def Hopf_dots_complex(t, z, mu, w0):
    F_complex = np.interp(t, tt, F_full_complex)
    return (mu + w0*1j - abs(z)**2)*z + F_complex


F = 0.01
mu = -0.1
w0 = 2*np.pi
num_cyc = 200
z0 = [0 + 0j]  #initial condition
dt = 0.001
pts_per_cycle = int(round(2*np.pi/(w0*dt)))
N     = num_cyc*pts_per_cycle
tt = np.linspace(-N*dt/2, N*dt/2, N)
tau_shift1 = 3
tau_shift2 = 1  #tau = d/v.  T=1/400 sec. tau/T = 400*1cm/(1 m/s)
tau_shift3 = 0.3
Phi1 = np.pi/( 1 + np.exp(-tt/tau_shift1) )
Phi2 = np.pi/( 1 + np.exp(-tt/tau_shift2) )
Phi3 = np.pi/( 1 + np.exp(-tt/tau_shift3) )
#Phi1 = np.pi*(  0.5*(tt/tau_shift1)/np.sqrt(1 + (tt/tau_shift1)**2 )  + 0.5  ) 
#Phi2 = np.pi*(  0.5*(tt/tau_shift2)/np.sqrt(1 + (tt/tau_shift2)**2 )  + 0.5  )
#Phi3 = np.pi*(  0.5*(tt/tau_shift3)/np.sqrt(1 + (tt/tau_shift3)**2 )  + 0.5  )
#Phi1 = np.pi*(  3*(tt/tau_shift1)/(1 + (tt/tau_shift1)**2 )  ) 
#Phi2 = np.pi*(  3*(tt/tau_shift2)/(1 + (tt/tau_shift2)**2 )  )
#Phi3 = np.pi*(  3*(tt/tau_shift3)/(1 + (tt/tau_shift3)**2 )  )


F_full_complex = F*np.exp(1j*(w0*tt + Phi1))
sol = solve_ivp(Hopf_dots_complex, (tt[0], tt[-1]), z0, args=(mu, w0), t_eval=tt, method='RK45')
x1 = np.real(sol.y[0])

F_full_complex = F*np.exp(1j*(w0*tt + Phi2))
sol = solve_ivp(Hopf_dots_complex, (tt[0], tt[-1]), z0, args=(mu, w0), t_eval=tt, method='RK45')
x2 = np.real(sol.y[0])

F_full_complex = F*np.exp(1j*(w0*tt + Phi3))
sol = solve_ivp(Hopf_dots_complex, (tt[0], tt[-1]), z0, args=(mu, w0), t_eval=tt, method='RK45')
x3 = np.real(sol.y[0])


x_limit = 30
y_limit = 0.15
fig = plt.figure(figsize=(10, 3))
plt.subplots_adjust(left=0.08, right=0.97, bottom=0.17, top=0.86, wspace=0.3)
ax1  = plt.subplot2grid((1, 3), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((1, 3), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((1, 3), (0,2), colspan=1, rowspan=1)
ax1.plot(tt, x1, color='black')
ax1.plot(tt, 0.22*(Phi1/np.pi - 1/2), "--", color='red')
ax1.set_xlim(-x_limit, x_limit)
ax1.set_ylim(-y_limit, y_limit)
ax1.set_xlabel("Time (cycles)")
ax1.set_ylabel(r'$\Re[z(t)]$')
ax2.plot(tt, x2, color='black')
ax2.plot(tt, 0.22*(Phi2/np.pi - 1/2), "--", color='red')
ax2.set_xlim(-x_limit, x_limit)
ax2.set_ylim(-y_limit, y_limit)
ax2.set_xlabel("Time (cycles)")
ax3.plot(tt, x3, color='black')
ax3.plot(tt, 0.22*(Phi3/np.pi - 1/2), "--", color='red')
ax3.set_xlim(-x_limit, x_limit)
ax3.set_ylim(-y_limit, y_limit)
ax3.set_xlabel("Time (cycles)")
ax1.text(-0.1, 1.15, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax2.text(-0.1, 1.15, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax3.text(-0.1, 1.15, "C", transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

plt.savefig("C:/Users/Justin/Desktop/Phase_shift.jpeg", dpi=300)


plt.show()



code.interact(local=locals())  #allows interaction with variables in terminal after



