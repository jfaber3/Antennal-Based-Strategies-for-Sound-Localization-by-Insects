import numpy as np
import matplotlib.pyplot as plt
import code

def FFT(x, dt):
	N = len(x)
	f = np.fft.fftfreq(N, d=dt)
	f = f[:int(N/2)]
	xf = np.fft.fft(x)
	xf = (2.0/N)*np.abs(xf[:int(N/2)])
	return f, xf


np.random.seed(35)  #43

N  = 1001
dt = 0.0001  #10k samp/sec
tt = np.linspace(-N*dt/2, N*dt/2, N)
v1 = 1    #velocity (m/sec)
v2 = 2
v3 = 5
v4 = 10
cc = 0.01  #clostest pass (meters) 
freq = 400  #Hz
noise_str = 0.1

#Noise
x1 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v1*tt)**2 )**(3/2) + np.random.normal(scale=noise_str, size=len(tt))
x2 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v2*tt)**2 )**(3/2) + np.random.normal(scale=noise_str, size=len(tt))
x3 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v3*tt)**2 )**(3/2) + np.random.normal(scale=noise_str, size=len(tt))
x4 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v4*tt)**2 )**(3/2) + np.random.normal(scale=noise_str, size=len(tt))
f, xf1 = FFT(x1, dt)
f, xf2 = FFT(x2, dt)
f, xf3 = FFT(x3, dt)
f, xf4 = FFT(x4, dt)

#No Noise
x1_0 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v1*tt)**2 )**(3/2)
x2_0 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v2*tt)**2 )**(3/2)
x3_0 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v3*tt)**2 )**(3/2)
x4_0 = (cc)**3 * np.cos(2*np.pi*freq*tt) / ( (cc)**2 + (v4*tt)**2 )**(3/2)
f, xf1_0 = FFT(x1_0, dt)
f, xf2_0 = FFT(x2_0, dt)
f, xf3_0 = FFT(x3_0, dt)
f, xf4_0 = FFT(x4_0, dt)


fig = plt.figure(figsize=(5, 4))
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.95, wspace=0.5, hspace=0.1)
ax1  = plt.subplot2grid((4, 2), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((4, 2), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((4, 2), (1,0), colspan=1, rowspan=1)
ax4  = plt.subplot2grid((4, 2), (1,1), colspan=1, rowspan=1)
ax5  = plt.subplot2grid((4, 2), (2,0), colspan=1, rowspan=1)
ax6  = plt.subplot2grid((4, 2), (2,1), colspan=1, rowspan=1)
ax7  = plt.subplot2grid((4, 2), (3,0), colspan=1, rowspan=1)
ax8  = plt.subplot2grid((4, 2), (3,1), colspan=1, rowspan=1)


ax1.plot(tt, x1, color='black')
ax1.plot(tt, x1_0, color='red', linewidth=1)
ax1.set_xlim(-0.02, 0.02)
ax1.set_ylim(-1.3, 1.3)
#ax1.set_ylabel("x(t)")
ax1.set_xticks([])
ax1.set_yticks([-1, 0, 1])

ax3.plot(tt, x2, color='black')
ax3.plot(tt, x2_0, color='red', linewidth=1)
ax3.set_xlim(-0.02, 0.02)
ax3.set_ylim(-1.3, 1.3)
#ax3.set_ylabel("x(t)")
ax3.set_xticks([])
ax3.set_yticks([-1, 0, 1])

ax5.plot(tt, x3, color='black')
ax5.plot(tt, x3_0, color='red', linewidth=1)
ax5.set_xlim(-0.02, 0.02)
ax5.set_ylim(-1.3, 1.3)
#ax5.set_ylabel("x(t)")
ax5.set_xticks([])
ax5.set_yticks([-1, 0, 1])

ax7.plot(tt, x4, color='black')
ax7.plot(tt, x4_0, color='red', linewidth=1)
ax7.set_xlim(-0.02, 0.02)
ax7.set_ylim(-1.3, 1.3)
ax7.set_ylabel("$v(t)$")
ax7.set_xticks([])
ax7.set_yticks([-1, 0, 1])


ax2.plot(f, xf1/max(xf1_0), "-", color='black')
ax2.plot(f, xf1_0/max(xf1_0), "-", color='red', linewidth=1)
ax2.set_xlim(0, 2*freq)
ax2.set_ylim(0, 1.5)
#ax2.set_ylabel(r'$|\tilde{x}(f)|$')
ax2.set_xticks([])
ax2.set_yticks([0, 1])

ax4.plot(f, xf2/max(xf2_0), "-", color='black')
ax4.plot(f, xf2_0/max(xf2_0), "-", color='red', linewidth=1)
ax4.set_xlim(0, 2*freq)
ax4.set_ylim(0, 1.5)
#ax4.set_ylabel(r'$|\tilde{x}(f)|$')
ax4.set_xticks([])
ax4.set_yticks([0, 1])

ax6.plot(f, xf3/max(xf3_0), "-", color='black')
ax6.plot(f, xf3_0/max(xf3_0), "-", color='red', linewidth=1)
ax6.set_xlim(0, 2*freq)
ax6.set_ylim(0, 1.5)
#ax6.set_ylabel(r'$|\tilde{x}(f)|$')
ax6.set_xticks([])
ax6.set_yticks([0, 1])

ax8.plot(f, xf4/max(xf4_0), "-", color='black')
ax8.plot(f, xf4_0/max(xf4_0), "-", color='red', linewidth=1)
ax8.set_xlim(0, 2*freq)
ax8.set_ylim(0, 1.5)
ax8.set_ylabel(r'$|\tilde{v}(f)|$')
ax8.set_xticks([])
ax8.set_yticks([0, 1])

ax1.text(0.01, 0.85, str(v1) + " m/s")
ax3.text(0.01, 0.85, str(v2) + " m/s")
ax5.text(0.01, 0.85, str(v3) + " m/s")
ax7.text(0.008, 0.85, str(v4) + " m/s")
ax2.text(600, 1.25, str(v1) + " m/s")
ax4.text(600, 1.25, str(v2) + " m/s")
ax6.text(600, 1.25, str(v3) + " m/s")
ax8.text(560, 1.25, str(v4) + " m/s")


ax7.set_xlabel("Time (sec)")
ax7.set_xticks([-0.02, 0.0, 0.02])
ax8.set_xlabel("Frequency (Hz)")
ax8.set_xticks([0, 400, 800])

ax1.text(-0.15, 1.22, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax2.text(-0.15, 1.22, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:/Users/Justin/Desktop/Amp_Env_Noise.jpeg', dpi=300)


plt.show()


code.interact(local=locals())  #allows interaction with variables in terminal after


