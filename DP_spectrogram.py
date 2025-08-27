import numpy as np
import matplotlib.pyplot as plt
import code
from scipy.integrate import solve_ivp
from scipy import signal

clrs = ["#FF1F5B","#FFC61E", "#009ADE", "#00CD6C", "#AF58BA", "#F28522"]

def Hopf_dots_complex(t, z, mu, w0):
    F_complex = np.interp(t, tt, F_full_complex)
    return (mu + w0*1j - abs(z)**2)*z + F_complex


mu = -0.1
w0 = 2*np.pi
num_cyc = 8000
num_cyc_cut = 100
z0 = [0 + 0j]   #[mu**0.5 + 0j]  #initial condition
Ff = 1
Fm = 1
dt = 0.01
pts_per_cycle = int(round(2*np.pi/(w0*dt)))
N     = num_cyc*pts_per_cycle
N_cut = num_cyc_cut*pts_per_cycle
print(N, N_cut)
tt = np.linspace(0, (N+N_cut-1)*dt, N+N_cut)
tt_mid = num_cyc_cut + num_cyc/2
time_scale =200
Wf = 0.96*w0  -  0.0048*w0*( (tt-tt_mid)/time_scale )/np.sqrt(1 + ((tt-tt_mid)/time_scale)**2)  #+- 1% shift (2% total shift)
Phi_f = np.cumsum( Wf*dt )
wm = 1.04*w0
F_female = Ff*np.exp(1j*Phi_f)
F_male   = Fm*np.exp(1j*wm*tt)

F_full_complex = F_female + F_male
sol = solve_ivp(Hopf_dots_complex, (0, num_cyc+num_cyc_cut), z0, args=(mu, w0), t_eval=tt, method='RK45')
z = sol.y[0]
z = z[N_cut:]
x = np.real(z)


plt.figure(figsize=(5, 5))
Pxx, freqs, t_bins, im = plt.specgram(x, NFFT=160*pts_per_cycle, Fs=1/dt, noverlap=0, scale='linear', 
                                        mode='magnitude', cmap='Greys', vmin=0, vmax=0.004)
delta_f = freqs[1]-freqs[0]
plt.plot(tt[:-N_cut], wm*np.ones(N)/(2*np.pi), "-", color=clrs[2])
plt.plot(tt[:-N_cut], Wf[N_cut:]/(2*np.pi), "-", color=clrs[2])
plt.plot(tt[:-N_cut], (2*Wf[N_cut:] - 1*wm)/(2*np.pi), "-", color=clrs[0])
plt.plot(tt[:-N_cut], (3*Wf[N_cut:] - 2*wm)/(2*np.pi), "-", color=clrs[0])
plt.plot(tt[:-N_cut], (4*Wf[N_cut:] - 3*wm)/(2*np.pi), "-", color=clrs[0])
plt.plot(tt[:-N_cut], (5*Wf[N_cut:] - 4*wm)/(2*np.pi), "-", color=clrs[0])
plt.plot(tt[:-N_cut], (6*Wf[N_cut:] - 5*wm)/(2*np.pi), "-", color=clrs[0])
plt.ylim(0.6, 1.1)
plt.xticks([0, 2000, 4000, 6000, 8000])
plt.xlabel("Time (cycles)")
plt.ylabel("Frequency (cycles$^{-1}$)")
plt.text(6000, 1.08, "$\omega_2$")
plt.text(6000, 0.99, "$\omega_1$")
plt.text(6000, 0.9, "$2\omega_1 - \omega_2$")
plt.text(6000, 0.81, "$3\omega_1 - 2\omega_2$")
plt.text(6000, 0.725, "$4\omega_1 - 3\omega_2$")
plt.text(6000, 0.64, "$5\omega_1 - 4\omega_2$")

plt.savefig("C:/Users/Justin/Desktop/DP_modulation.jpeg", dpi=300)

plt.show()

code.interact(local=locals())  #allows interaction with variables in terminal after



'''



dt = 0.01*2*np.pi            #0.001*2*np.pi for 1000 pts per cycle,    dt*w0 = 0.001*2pi,     use 0.01*2pi as shortcut
num_cyc = 10000  #10k or 100k
num_cut = 500
w0 = 1.0
mu = -0.1
F  = 5.0
Dw = 0.01
w1 = w0 - Dw/2
w2 = w0 + Dw/2
mod_range = 0.001
mod_w = 3*w0/num_cyc   #mod_range/2     #limit is mod_range/2      2*w0/num_cyc gives 2 cycles

N     = int(round( num_cyc*2*np.pi/(w0*dt) ))
N_cut = int(round( num_cut*2*np.pi/(w0*dt) ))
print(N, N_cut)
tt = np.linspace(0, (N+N_cut-1)*dt, N+N_cut)
Fx1, Fy1, W = sin_FM_stim(tt, F, w1, mod_range, mod_w)
Fx2, Fy2 = F*np.cos(w2*tt),  F*np.sin(w2*tt)
x, y = HOPF(mu, w0, Fx1 + Fx2, Fy1 + Fy2, dt)
x, y = x[N_cut:], y[N_cut:]
f, xf = FFT(x, 2*np.pi/dt)





plt.figure(figsize=(6, 4))
Pxx, freqs, t_bins, im = plt.specgram(x, NFFT=100000, Fs=2*np.pi/dt, noverlap=0, scale='linear', mode='magnitude', cmap='Greys', vmin=0, vmax=0.02)
delta_f = freqs[1]-freqs[0]
plt.plot(tt/(2*np.pi), W + delta_f/2, "--", color='red')
plt.ylim(0.93, 1.01)
plt.xticks([0, 10000, 20000, 30000, 40000])
plt.axhline(y=w2 + delta_f/2, ls='dashed', color='red')
plt.xlabel("Time (cycles)")
plt.ylabel("Frequency (Hz)")
plt.xlim(0, num_cyc)
plt.show()
'''

#plt.savefig(r'C:\Users\Justin\Desktop\Fig3.jpeg', dpi=300)




