import random
import numpy as np
import matplotlib.pyplot as plt

def FFT(x, dt):
	N = len(x)
	f = np.fft.fftfreq(N, d=dt)
	f = f[:int(N/2)]
	xf = np.fft.fft(x)
	xf = (2.0/N)*np.abs(xf[:int(N/2)])
	return f, xf






N  = 1000
dt = 0.01
tt = np.linspace(0, (N-1)*dt, N)
t_mid = tt[int(N/2)]

env_width1 = 1000   #number of cycles
env_width2 = 2   #number of cycles
env_width3 = 1   #number of cycles
env_width4 = 0.5   #number of cycles

x1 = np.sin(2*np.pi*tt)*np.e**(-0.5*( (tt-t_mid)/env_width1)**2 )
x2 = np.sin(2*np.pi*tt)*np.e**(-0.5*( (tt-t_mid)/env_width2)**2 )
x3 = np.sin(2*np.pi*tt)*np.e**(-0.5*( (tt-t_mid)/env_width3)**2 )
x4 = np.sin(2*np.pi*tt)*np.e**(-0.5*( (tt-t_mid)/env_width4)**2 )
f, xf1 = FFT(x1, dt)
f, xf2 = FFT(x2, dt)
f, xf3 = FFT(x3, dt)
f, xf4 = FFT(x4, dt)


fig = plt.figure(figsize=(5, 6))
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, wspace=0.5, hspace=0.1)
ax1  = plt.subplot2grid((4, 2), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((4, 2), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((4, 2), (1,0), colspan=1, rowspan=1)
ax4  = plt.subplot2grid((4, 2), (1,1), colspan=1, rowspan=1)
ax5  = plt.subplot2grid((4, 2), (2,0), colspan=1, rowspan=1)
ax6  = plt.subplot2grid((4, 2), (2,1), colspan=1, rowspan=1)
ax7  = plt.subplot2grid((4, 2), (3,0), colspan=1, rowspan=1)
ax8  = plt.subplot2grid((4, 2), (3,1), colspan=1, rowspan=1)


ax1.plot(tt, x1, color='black')
ax1.set_ylim(-1.1, 1.1)
ax1.set_ylabel("x(t)")
ax1.set_xticks([])
ax1.set_yticks([-1, 0, 1])

ax3.plot(tt, x2, color='black')
ax3.set_ylim(-1.1, 1.1)
ax3.set_ylabel("x(t)")
ax3.set_xticks([])
ax3.set_yticks([-1, 0, 1])

ax5.plot(tt, x3, color='black')
ax5.set_ylim(-1.1, 1.1)
ax5.set_ylabel("x(t)")
ax5.set_xticks([])
ax5.set_yticks([-1, 0, 1])

ax7.plot(tt, x4, color='black')
ax7.set_ylim(-1.1, 1.1)
ax7.set_ylabel("x(t)")
ax7.set_xticks([])
ax7.set_yticks([-1, 0, 1])



ax2.plot(f, xf1, "o-", color='black', markersize=4)
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 1.1)
ax2.set_ylabel(r'$|\tilde{x}(f)|$')
ax2.set_xticks([])
ax2.set_yticks([0, 1])

ax4.plot(f, xf2, "o-", color='black', markersize=4)
ax4.set_xlim(0, 2)
ax4.set_ylim(0, 1.1)
ax4.set_ylabel(r'$|\tilde{x}(f)|$')
ax4.set_xticks([])
ax4.set_yticks([0, 1])

ax6.plot(f, xf3, "o-", color='black', markersize=4)
ax6.set_xlim(0, 2)
ax6.set_ylim(0, 1.1)
ax6.set_ylabel(r'$|\tilde{x}(f)|$')
ax6.set_xticks([])
ax6.set_yticks([0, 1])

ax8.plot(f, xf4, "o-", color='black', markersize=4)
ax8.set_xlim(0, 2)
ax8.set_ylim(0, 1.1)
ax8.set_ylabel(r'$|\tilde{x}(f)|$')
ax8.set_xticks([])
ax8.set_yticks([0, 1])




ax7.set_xlabel("Time (cycles)")
ax7.set_xticks([0, 5, 10])
ax8.set_xlabel("Frequency (Hz)")
ax8.set_xticks([0, 1, 2])

ax1.text(-0.12, 1.2, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.12, 1.2, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig1.jpeg', dpi=300)




