import numpy as np
import matplotlib.pyplot as plt

x, y            = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
x_tiny, y_tiny = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
Theta = np.arctan(x/y)
Theta_tiny = np.arctan(x_tiny/y_tiny)

u = 3*np.cos(Theta)*np.sin(Theta)
v = 3*np.cos(Theta)**2 - 1
A = (3*np.cos(Theta_tiny)**2 - 1)/(x_tiny**2 + y_tiny**2)**0.5  #not the correct r dependence but just for illustration




fig = plt.figure(figsize=(10, 7))
plt.subplots_adjust(left=0.08, right=0.95, bottom=0.05, top=0.93, wspace=0.25, hspace=0.25)
ax1  = plt.subplot2grid((2, 2), (0,0), colspan=1, rowspan=1)
ax2  = plt.subplot2grid((2, 2), (0,1), colspan=1, rowspan=1)
ax3  = plt.subplot2grid((2, 2), (1,0), colspan=1, rowspan=1)
ax4  = plt.subplot2grid((2, 2), (1,1), colspan=1, rowspan=1)

ax1.imshow(A, cmap='coolwarm', interpolation='bilinear', extent=(-10, 10, -10, 10), vmin=-0.5, vmax=1)
ax1.quiver(x, y, u/(u**2 + v**2)**0.5, v/(u**2 + v**2)**0.5, color='black', pivot='middle')
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_xticks([])
ax1.set_yticks([])


tt = np.linspace(-20, 20, 100000)
w = 30
V = 1
c = 20  #speed of sound
clr1 = 'magenta'
d1 = 4.2
y1 = d1*np.ones(len(tt))
x1 = V*tt
T1 = np.arctan(x1/y1)
R1 = (x1**2 + y1**2)**0.5
A1 = (3*(np.cos(T1)**2) + 1)/R1**3
ax1.plot(x1, y1, "--", color=clr1)
ax2.plot(tt, A1*np.sin(w*tt), color=clr1)
clr2 = 'blue'
d2 = 2.3  #2.3
y2 = -V*tt
x2 = d2*np.ones(len(tt))
T2 = np.arctan(x2/y2)
R2 = (x2**2 + y2**2)**0.5
A2 = (3*(np.cos(T2)**2) + 1)/R2**3
ax1.plot(x2, y2, "--", color=clr2)
ax2.plot(tt, A2*np.sin(w*tt) + 0.35, color=clr2)
clr3 = 'springgreen'
d3 = 5
y3 = -V*tt*(1/2)**0.5
x3 =  V*tt*(1/2)**0.5 + d3
T3 = np.arctan(x3/y3)
R3 = (x3**2 + y3**2)**0.5
A3 = (3*(np.cos(T3)**2) + 1)/R3**3
ax1.plot(x3, y3, "--", color=clr3)
ax2.plot(tt + 5, A3*np.sin(w*tt) + 0.15, color=clr3)
ax2.set_xlim(-10, 10)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel("Time")
ax2.set_ylabel("Velocity")

R1_dot = (R1[2:] - R1[:-2])/(tt[2]-tt[0])
R2_dot = (R2[2:] - R2[:-2])/(tt[2]-tt[0])
R3_dot = (R3[2:] - R3[:-2])/(tt[2]-tt[0])
f1 = 1 - R1_dot/c
f2 = 1 - R2_dot/c
f3 = 1 - R3_dot/c
ax3.axhline(y=1, color='black', ls='dotted')
ax3.plot(tt[1:-1], f1, clr1)
ax3.plot(tt[1:-1], f2, clr2)
ax3.plot(tt[1:-1] + 3.5, f3, clr3)
ax3.set_xlim(-15, 15)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlabel("Time")
ax3.set_ylabel("Frequency")



u1, v1 = 3*np.cos(T1)*np.sin(T1), 3*np.cos(T1)**2 - 1
u2, v2 = 3*np.cos(T2)*np.sin(T2), 3*np.cos(T2)**2 - 1
u3, v3 = 3*np.cos(T3)*np.sin(T3), 3*np.cos(T3)**2 - 1
angle1 = np.unwrap(np.arctan2(u1, v1)) - np.pi/2
angle2 = np.unwrap(np.arctan2(u2, v2))
angle3 = np.unwrap(np.arctan2(u3, v3)) - np.pi/2 - np.pi/4  #with respect to direction of flight
angle1 = (angle1 + 2*np.pi)%np.pi
angle2 = (angle2 + 0)%np.pi
angle3 = (angle3 + 2*np.pi)%np.pi
ax4.plot(tt, angle1, "o", markersize=0.5, color=clr1)
ax4.plot(tt, angle2, "o", markersize=0.5, color=clr2)
ax4.plot(tt, angle3, "o", markersize=0.5, color=clr3)
ax4.set_xlim(tt[0], tt[-1])
ax4.set_ylim(0, np.pi)
ax4.set_yticks([0, np.pi/2, np.pi])
ax4.set_xticks([])
ax4.set_yticklabels([r'0', r'$\frac{\pi}{2}$', r'$\pi$'])
ax4.set_xlabel("Time")
ax4.set_ylabel("Angle")

ax1.text(-0.27, 1.1, "A", transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax2.text(-0.07, 1.1, "B", transform=ax2.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax3.text(-0.07, 1.1, "C", transform=ax3.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')
ax4.text(-0.07, 1.1, "D", transform=ax4.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

#plt.savefig(r'C:\Users\Justin\Desktop\Fig1.jpeg', dpi=300)




