import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

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




#MODULATED SIGNAL FUNCTIONS

def move_avg(x, window):
    if window%2 != 1:
        print("window must be odd")
        window += 1
    filt_x = np.zeros(len(x), dtype=float)
    for i in range(int(window/2),  len(x) - int(window/2)):  # filter middle of data
        filt_x[i] = sum(x[i - int(window/2):i + int(window/2) + 1])/window
    for i in range(int(window/2)):  # filter ends of data
        filt_x[i] = sum(x[0:(i + int(window/2))]) / len(x[0:(i + int(window/2))])
        filt_x[len(x) - 1 - i] = sum(x[(len(x) - i - int(window/2)):len(x)]) / len(x[(len(x) - i - int(window/2)):len(x)])
    return filt_x

def FFT(x, framerate):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff = (2.0/N)*np.abs(xf[0:int(N/2)])
    return f, xff

def PSD(x, framerate):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    xff = (2.0/(N*framerate))*np.abs(xf[0:int(N/2)])**2
    return f, xff

def LP_filter(x, framerate, cutoff_f):
    N = len(x)
    xf = np.fft.fft(x)
    f = np.linspace(0.0, framerate/2, int(N/2))
    for i in range(len(f)):
        if f[i] > cutoff_f:
            xf[i] = 0.0 + 0.0j
            xf[len(xf)-i-1] = 0.0 + 0.0j
    return np.real(np.fft.ifft(xf))

def generate_FM_signal(dt, N, F, w, w_dev):
    Dw = np.random.normal(0, w_dev, N)
    Dw = LP_filter(Dw, 2*np.pi/dt, 1*w)
    Dw *= w_dev/Dw.std()
    phase = np.zeros(N)
    phase[0] = 2*np.pi*random.random()  #-np.pi/2
    for i in range(1, len(phase)):
        phase[i] = phase[i-1] + (w + Dw[i])*dt
    return F*np.cos(phase), F*np.sin(phase)

def Freq_Sweep_Mask(total_power, w0, w_dev_range, w_sweep, N, dt):   #linearly increasing freq sweep
    Ws = np.zeros(N)
    Ws[0] = w0 - w_dev_range
    sweep_rate = (w_sweep/w0) * 2*w_dev_range/(2*np.pi/w0)  #(rad/s)/s
    for i in range(1, len(Ws)):
        Ws[i] = Ws[i-1] + sweep_rate*dt
        if Ws[i] > w0 + w_dev_range:
            Ws[i] = w0 - w_dev_range
    phases = np.zeros(N)
    phases[0] = 2*np.pi*random.random()
    for i in range(1, len(phases)):
        phases[i] = phases[i-1] + Ws[i]*dt
    x, y = np.cos(phases), np.sin(phases)
    x *= ((total_power/2)**0.5)/x.std()
    y *= ((total_power/2)**0.5)/y.std()
    return x, y

def sin_FM_stim(tt, F, w_center, mod_range, mod_w):
    W = mod_range * np.sin(mod_w*tt) + w_center
    phase = np.zeros(len(tt))
    dt = tt[1] - tt[0]
    phase[0] = 0  #2*np.pi*random.random()  #-np.pi/2
    for i in range(1, len(phase)):
        phase[i] = phase[i-1] + W[i-1]*dt
    return F*np.cos(phase), F*np.sin(phase), W



dt = 0.01*2*np.pi            #0.001*2*np.pi for 1000 pts per cycle,    dt*w0 = 0.001*2pi,     use 0.01*2pi as shortcut
num_cyc = 40000  #10k or 100k
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


'''
plt.figure()
plt.plot(f, xf)
plt.xlim(0.9, 1.1)
plt.ylim(0.01, 10)
plt.yscale("log")

window = 100000
plt.figure()
for i in range(0, 10):
    f, xf = FFT(x[i*window:(i+1)*window], 2*np.pi/dt)
    plt.plot(f, xf)
plt.xlim(0.9, 1.1)
plt.ylim(0.01, 10)
plt.yscale("log")
'''


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


#plt.savefig(r'C:\Users\Justin\Desktop\Fig3.jpeg', dpi=300)




