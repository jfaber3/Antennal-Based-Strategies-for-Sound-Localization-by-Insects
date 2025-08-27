import numpy as np
import matplotlib.pyplot as plt
import code

x, y            = np.meshgrid(np.linspace(-1, 1, 20),  np.linspace(-1, 1, 20))
x_tiny, y_tiny  = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Theta = np.arctan2(x, y)
Theta_tiny = np.arctan2(x_tiny, y_tiny)
r = (x**2 + y**2)**0.5
r_tiny = (x_tiny**2 + y_tiny**2)**0.5
v_mag = (1/r_tiny**3)*(3*np.cos(Theta_tiny)**2 + 1)**0.5
u = 3*np.cos(Theta)*np.sin(Theta)
v = 3*np.cos(Theta)**2 - 1



fig = plt.figure(figsize=(10, 4.5))
plt.subplots_adjust(left=0.05, right=0.97, bottom=0.12, top=0.93, wspace=0.5, hspace=0.4)
ax1  = plt.subplot2grid((2, 4), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((2, 4), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((2, 4), (0,2), colspan=1, rowspan=1)
ax4  = plt.subplot2grid((2, 4), (0,3), colspan=1, rowspan=1)
ax5  = plt.subplot2grid((2, 4), (1,0), colspan=1, rowspan=1)
ax6  = plt.subplot2grid((2, 4), (1,1), colspan=1, rowspan=1)
ax7  = plt.subplot2grid((2, 4), (1,2), colspan=1, rowspan=1)
ax8  = plt.subplot2grid((2, 4), (1,3), colspan=1, rowspan=1)


v_mag_log = np.log10(v_mag)
ax1.imshow(v_mag_log, cmap='Greys', interpolation='bilinear', extent=(-1, 1, -1, 1), 
            vmin=np.quantile(v_mag_log, 0.5), vmax=np.quantile(v_mag_log, 0.99))
ax1.quiver(x, y, u/(u**2 + v**2)**0.5, v/(u**2 + v**2)**0.5, color='black', pivot='middle', scale=22, width=0.004)
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_xticks([-1, 0, 1], labels=["-1", "0", "1"])
ax1.set_yticks([-1, 0, 1], labels=["-1", "0", "1"])
ax1.set_xlabel("Position (m)")
ax1.set_ylabel("Position (m)")

ax5.imshow(v_mag_log, cmap='Greys', interpolation='bilinear', extent=(-1, 1, -1, 1), 
            vmin=np.quantile(v_mag_log, 0.5), vmax=np.quantile(v_mag_log, 0.99))
ax5.quiver(x, y, u/(u**2 + v**2)**0.5, v/(u**2 + v**2)**0.5, color='black', pivot='middle', scale=22, width=0.004)
ax5.set_xlim(-1, 1)
ax5.set_ylim(-1, 1)
ax5.set_xticks([-1, 0, 1], labels=["-1", "0", "1"])
ax5.set_yticks([-1, 0, 1], labels=["-1", "0", "1"])
ax5.set_xlabel("Position (m)")
ax5.set_ylabel("Position (m)")




clrs = ["#FF1F5B","#FFC61E", "#009ADE", "#00CD6C", "#AF58BA", "#F28522"]
tt = np.linspace(-1, 1, 100000) #time in seconds
f = 400  #sound frequency (Hz)
V = 2   #relative insect speed (m/s)
c = 343  #speed of sound (m/s)
t_span = 0.6

d1 = 0.3
y1 = d1*np.ones(len(tt))
x1 = V*tt
T1 = np.arctan2(x1, y1)
R1 = (x1**2 + y1**2)**0.5
A1 = np.sqrt(3*(np.cos(T1)**2) + 1)/R1**3
ax1.plot(x1, y1, color=clrs[0])
ax2.plot(tt, A1,         color=clrs[0])
ax3.plot(tt, A1/max(A1), color=clrs[0])

d2 = 0.3
y2 = -V*tt
x2 = d2*np.ones(len(tt))
T2 = np.arctan2(x2, y2)
R2 = (x2**2 + y2**2)**0.5
A2 = np.sqrt(3*(np.cos(T2)**2) + 1)/R2**3
ax1.plot(x2, y2, color=clrs[1])
ax2.plot(tt, A2, color=clrs[1])
ax3.plot(tt, A2/max(A2), color=clrs[1])

d3 = 0.3
y3 = -V*tt*(1/2)**0.5 + np.sqrt(1/2)*d3
x3 =  V*tt*(1/2)**0.5 + np.sqrt(1/2)*d3
T3 = np.arctan2(x3, y3)
R3 = (x3**2 + y3**2)**0.5
A3 = np.sqrt(3*(np.cos(T3)**2) + 1)/R3**3
t_offset3 = tt[np.argmax(A3)]
ax1.plot(x3, y3, color=clrs[2])
ax2.plot(tt - t_offset3, A3, color=clrs[2])
ax3.plot(tt - t_offset3, A3/max(A3), color=clrs[2])

d4 = 0.1
y4 = -V*tt*(1/2)**0.5 + np.sqrt(1/2)*d4
x4 =  V*tt*(1/2)**0.5 + np.sqrt(1/2)*d4
T4 = np.arctan2(x4, y4)
R4 = (x4**2 + y4**2)**0.5
A4 = np.sqrt(3*(np.cos(T3)**2) + 1)/R4**3
t_offset4 = tt[np.argmax(A4)]
ax5.plot(x4, y4, color=clrs[3])
ax6.plot(tt - t_offset4, A4, color=clrs[3])
ax7.plot(tt - t_offset4, A4/max(A4), color=clrs[3])

d5 = 0.3
y5 = -V*tt*(1/2)**0.5 + np.sqrt(1/2)*d5
x5 =  V*tt*(1/2)**0.5 + np.sqrt(1/2)*d5
T5 = np.arctan2(x5, y5)
R5 = (x5**2 + y5**2)**0.5
A5 = np.sqrt(3*(np.cos(T5)**2) + 1)/R5**3
t_offset5 = tt[np.argmax(A5)]
ax5.plot(x5, y5, color=clrs[4])
ax6.plot(tt - t_offset5, A5, color=clrs[4])
ax7.plot(tt - t_offset5, A5/max(A5), color=clrs[4])

d6 = 0.5
y6 = -V*tt*(1/2)**0.5 + np.sqrt(1/2)*d6
x6 =  V*tt*(1/2)**0.5 + np.sqrt(1/2)*d6
T6 = np.arctan2(x6, y6)
R6 = (x6**2 + y6**2)**0.5
A6 = np.sqrt(3*(np.cos(T6)**2) + 1)/R6**3
t_offset6 = tt[np.argmax(A6)]
ax5.plot(x6, y6, color=clrs[5])
ax6.plot(tt - t_offset6, A6, color=clrs[5])
ax7.plot(tt - t_offset6, A6/max(A6), color=clrs[5])

ax2.set_yscale("log")
ax3.set_yscale("log")
ax6.set_yscale("log")
ax7.set_yscale("log")
ax2.set_ylim(0.2, 200)
ax3.set_ylim(0.002, 2)
ax6.set_ylim(0.3, 3000)
ax7.set_ylim(0.0001, 2)


R1_dot = (R1[2:] - R1[:-2])/(tt[2]-tt[0])
R2_dot = (R2[2:] - R2[:-2])/(tt[2]-tt[0])
R3_dot = (R3[2:] - R3[:-2])/(tt[2]-tt[0])
R4_dot = (R4[2:] - R4[:-2])/(tt[2]-tt[0])
R5_dot = (R5[2:] - R5[:-2])/(tt[2]-tt[0])
R6_dot = (R6[2:] - R6[:-2])/(tt[2]-tt[0])
f1 = f - (c/f)*R1_dot
f2 = f - (c/f)*R2_dot
f3 = f - (c/f)*R3_dot
f4 = f - (c/f)*R4_dot
f5 = f - (c/f)*R5_dot
f6 = f - (c/f)*R6_dot
ax4.axhline(y=f, color='black', ls='dotted')
ax4.plot(tt[1:-1], f1, clrs[0])
ax4.plot(tt[1:-1], f2, clrs[1])
ax4.plot(tt[1:-1], f3, clrs[2])
ax8.axhline(y=f, color='black', ls='dotted')
ax8.plot(tt[1:-1], f4, clrs[3])
ax8.plot(tt[1:-1], f5, clrs[4])
ax8.plot(tt[1:-1], f6, clrs[5])

ax2.set_xlim(-t_span, t_span)
ax2.set_xlabel("Time (sec)")
ax2.set_ylabel("Amplitude (arb. unit)")
ax3.set_xlim(-t_span, t_span)
ax3.set_xlabel("Time (sec)")
ax3.set_ylabel("Scaled amplitude")
ax4.set_xlim(-t_span, t_span)
ax4.set_xlabel("Time (sec)")
ax4.set_ylabel("Frequency (Hz)")

ax6.set_xlim(-t_span, t_span)
ax6.set_xlabel("Time (sec)")
ax6.set_ylabel("Amplitude (arb. unit)")
ax7.set_xlim(-t_span, t_span)
ax7.set_xlabel("Time (sec)")
ax7.set_ylabel("Scaled amplitude")
ax8.set_xlim(-t_span, t_span)
ax8.set_xlabel("Time (sec)")
ax8.set_ylabel("Frequency (Hz)")


ax1.text(-0.09, 1.17, "A", transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax2.text(-0.09, 1.17, "B", transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax3.text(-0.09, 1.17, "C", transform=ax3.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax4.text(-0.09, 1.17, "D", transform=ax4.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax5.text(-0.09, 1.17, "E", transform=ax5.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax6.text(-0.09, 1.17, "F", transform=ax6.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax7.text(-0.09, 1.17, "G", transform=ax7.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
ax8.text(-0.09, 1.17, "H", transform=ax8.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')

#plt.savefig("C:/Users/Justin/Desktop/Fig1.jpeg", dpi=300)
plt.show()


code.interact(local=locals())  #allows interaction with variables in terminal after






