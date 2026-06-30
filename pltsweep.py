import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import stft


def dealIQ(t, I_raw_matrix, Q_raw_matrix):
    dt = t[1] - t[0]         # Automatic calculation of time step (s)
    fs = 1 / dt             # Sampling frequency (Hz)
    cutoff_freq = 20.0e9      # Cutoff frequency set to 3 GHz
    # Normalized cutoff frequency (Wn is relative to Nyquist frequency fs/2)
    Wn = cutoff_freq / (fs / 2)
    # Design a 4th-order lowpass Butterworth filter
    b, a = butter(4, Wn, btype='low')
    I_filtered = np.zeros_like(I_raw_matrix)
    Q_filtered = np.zeros_like(Q_raw_matrix)
    # Apply zero-phase forward-backward filtering across all channels
    for ch in range(I_raw_matrix.shape[0]):
        I_filtered[ch, :] = filtfilt(b, a, I_raw_matrix[ch, :])
        Q_filtered[ch, :] = filtfilt(b, a, Q_raw_matrix[ch, :])
    return I_filtered, Q_filtered


def X_refraction_index_style(f, Bt, ne):

    # 严格采用 MATLAB 代码中的物理常数
    e = 1.602 * 10**(-19)
    epsilon0 = 8.854 * 10**(-12)
    me = 9.1096 * 10**(-31)
    
    Wpe = np.sqrt(e**2 * ne / (epsilon0 * me))
    Wce = e * Bt / me

    w = 2.0 * np.pi * f
    X = (Wpe**2) / (w**2)
    Y = Wce / w
    
    with np.errstate(divide='ignore', invalid='ignore'):
        N1 = 1.0 - (X * (1.0 - X)) / (1.0 - X - Y**2)
        N1 = np.where((N1 < 0) | np.isnan(N1), 0.0, N1)
        
    N = np.sqrt(N1)
    return N

def find_ne(f, Bt):
    """
     1 - X - Y^2 - X*Y = 0 -> f = f_ce/2 + sqrt((f_ce/2)^2 + f_pe^2)
    """
    e = 1.602 * 10**(-19)
    epsilon0 = 8.854 * 10**(-12)
    me = 9.1096 * 10**(-31)
    
    plasma_constant = (4.0 * np.pi**2 * epsilon0 * me) / (e**2)
    f_ce = (e * Bt) / (2.0 * np.pi * me)
    
    cutoff_term = f**2 - f * f_ce
    ne = plasma_constant * cutoff_term
    return np.where(ne < 0, 0.0, ne)


def refl_inversion_Xmode(f0, x0, n0, freq, phase, B0, R0):
    c = 3.0 * 10**8  # 严格对齐 MATLAB 的光速值
    
    x1 = []
    f1 = []
    phi = [0.0]
    dph = [0.0]
    
    for k in range(len(phase)):
        x = np.array([x0] + x1)
        f = np.array([f0] + f1)
        
        Bt1 = B0 * R0 / x 
        n1 = find_ne(f, Bt1)
        N_temp = X_refraction_index_style(freq[k], Bt1, n1)
        
        if len(x) == 1:
            phi_temp = 0.0
        else:
            phi_temp = -np.trapezoid((4.0 * np.pi * freq[k] / c) * N_temp, x)

        dphi = phase[k] - phi_temp
        EPSILON = 1e-6  

        N_last = N_temp[-1]
        if abs(N_last) < EPSILON:
            N_last = EPSILON if N_last >= 0 else -EPSILON

        x_temp = x[-1] - (2.0 * dphi * c / (4.0 * np.pi * freq[k])) / N_last
        #x_temp = x[-1] - (2.0 * dphi * c / (4.0 * np.pi * freq[k])) / N_temp[-1]
        
        phi.append(phi_temp)
        dph.append(dphi)
        x1.append(x_temp)
        f1.append(freq[k])
        
    x2 = np.array([x0] + x1)
    f2 = np.array([f0] + list(freq))
    
    Bt2 = B0 * R0 / x2
    n2 = find_ne(f2, Bt2)
    
    x3 = np.array([x0] + list((x2[:-1] + x2[1:]) / 2.0))
    n3 = np.abs(np.array([n0] + list((n2[:-1] + n2[1:]) / 2.0)))
    
    return x3, n3

def get_f_ce(B0,R0,r_pos):
    e = 1.602 * 10**(-19)
    epsilon0 = 8.854 * 10**(-12)
    me = 9.1096 * 10**(-31)
    B_r = B0 * R0 / (r_pos) 
    f_ce = (e * B_r) / (2.0 * np.pi * me)
    return f_ce


grdata=np.loadtxt('gr.dat')




f_start, f_end, T_sweep = grdata[5],grdata[6],grdata[7]

ch = 0
window_length = 20001

t_start = 56 #ns
R_boundary = 2.368 #fix R0
R_inner_wall = grdata[0]

ave=20000


fce = get_f_ce(grdata[3],grdata[2],R_boundary)
print('fce0 = ',fce/1e9)

c = 3e8
time0 = np.loadtxt('time.dat')
IQ_I_raw = np.loadtxt('IQ_Isig.dat')
IQ_Q_raw = np.loadtxt('IQ_Qsig.dat')
E_rece = np.loadtxt('receive_E.dat')
amplitude = np.abs(E_rece)

IQ_I_vacuum_raw = np.loadtxt('IQ_Isig_vacuum.dat')
IQ_Q_vacuum_raw = np.loadtxt('IQ_Qsig_vacuum.dat')

if IQ_I_raw.ndim == 1:
    IQ_I_raw = IQ_I_raw.reshape(1, -1)
    IQ_Q_raw = IQ_Q_raw.reshape(1, -1)
    IQ_I_vacuum_raw = IQ_I_vacuum_raw.reshape(1, -1)
    IQ_Q_vacuum_raw = IQ_Q_vacuum_raw.reshape(1, -1)


IQ_I_raw, IQ_Q_raw = dealIQ(time0/1e9, IQ_I_raw, IQ_Q_raw)
IQ_I_vacuum_raw, IQ_Q_vacuum_raw = dealIQ(time0/1e9, IQ_I_vacuum_raw, IQ_Q_vacuum_raw)

IQ_I_raw = IQ_I_raw[ch, :]
IQ_Q_raw = IQ_Q_raw[ch, :]
IQ_I_vacuum_raw = IQ_I_vacuum_raw[ch, :]
IQ_Q_vacuum_raw = IQ_Q_vacuum_raw[ch, :]

A_complex_vacuum = IQ_I_vacuum_raw + 1j * IQ_Q_vacuum_raw
A_complex_plasma = IQ_I_raw+ 1j * IQ_Q_raw

phi_plasma = np.unwrap(np.angle(A_complex_plasma))
phi_vacuum = np.unwrap(np.angle(A_complex_vacuum))
f_beat_instantaneous = savgol_filter(phi_plasma, 
                             window_length=window_length, 
                             polyorder=3, 
                             deriv=1, 
                             delta=(2.0 * np.pi * (time0[1]-time0[0])/1e9))


fb = savgol_filter(f_beat_instantaneous, window_length=window_length, polyorder=2)


time_s = time0 / 1e9                 
chirp_rate = (f_end - f_start) / T_sweep
freq_axis_full0 = f_start + chirp_rate * time_s
freq_axis_full = freq_axis_full0-fb/2

idx_time = np.where(time0 > t_start)[0]

time = time_s[idx_time]
freq_axis = freq_axis_full[idx_time]
phi_plasma_rel = phi_plasma[idx_time]
phi_vacuum_aligned = np.interp(freq_axis, freq_axis_full0, phi_vacuum)
phi_net = phi_plasma_rel - phi_vacuum_aligned



#phi_net = savgol_filter(phi_net,  window_length=window_length, polyorder=3)

f0 = freq_axis[0]
fb0 = fb[idx_time]
fb0 = fb0[0]
L_extra = R_boundary - R_inner_wall-grdata[4]
phi_vacuum_compensate = (4.0 * np.pi * L_extra / c) * (freq_axis - f0)
phi_rx_net = phi_net + phi_vacuum_compensate



df = freq_axis_full0[2]-freq_axis_full0[1]
dphi_df_vac = savgol_filter(phi_vacuum, 
                             window_length=window_length, 
                             polyorder=3, 
                             deriv=1, 
                             delta=df)
tau_vac_profile = dphi_df_vac / (2.0 * np.pi)

tau_f0 = fb0/chirp_rate
tau_vac = np.mean(tau_vac_profile[int(len(tau_vac_profile)/2) : -2000])
dtau = tau_vac - tau_f0
dL = c*dtau/2
Redge = R_inner_wall+dL-grdata[4]
print('tau_vac=',tau_vac*1e9,' tau_f0=',tau_f0*1e9,' Redge=',Redge)

tau_total = fb / chirp_rate


idx1=0
phi_rx2=[]
freq_axis2 = []
time2=[]
fb2=[]
while idx1+ave < len(phi_rx_net):
    phi_rx2.append(np.mean(phi_rx_net[idx1:idx1+ave]))
    freq_axis2.append(np.mean(freq_axis[idx1:idx1+ave]))
    time2.append(np.mean(time[idx1:idx1+ave]))
    fb2.append(np.mean(fb[idx1:idx1+ave]))
    idx1 += ave

phi_rx_net=np.array(phi_rx2)
freq_axis=np.array(freq_axis2)
time=np.array(time2)

phi_rx_net = phi_rx_net-np.min(phi_rx_net)

B0_magnetic_axis = grdata[3]   
R0_major_radius = grdata[2]



R, ne = refl_inversion_Xmode(
    f0, 
    R_boundary, 
    0.0, 
    freq_axis[1:], 
    phi_rx_net[1:], 
    B0=B0_magnetic_axis, 
    R0=R0_major_radius
)



freq_axis = np.array(freq_axis)
ne_2d_0 = np.loadtxt('ne_2d.dat')
R_0 = np.loadtxt('R.dat')
[nx,ny] = np.shape(ne_2d_0)
mid = int(ny / 2)
plt.figure(figsize=(5, 10))
plt.subplot(411)
plt.plot(time0,amplitude)
plt.ylabel('Erece')
plt.grid(True, alpha=0.3)
plt.xlim([0,np.max(time0)])
plt.subplot(412)
plt.plot(time*1e9,phi_rx_net,linewidth=2)
plt.ylabel('Phase (rad)')
plt.xlim([0,np.max(time0)])
plt.grid(True, alpha=0.3)
plt.subplot(413)
plt.plot(time0,fb/1e9,linewidth=2)
plt.grid(True, alpha=0.3)
plt.ylim([0,np.max(fb/1e9)])
plt.ylabel('fb (GHz)')
plt.grid(True, alpha=0.3)
plt.xlim([0,np.max(time0)])
plt.subplot(414)
plt.plot(time0,tau_total*1e9,linewidth=2)
plt.ylim([0,np.max(tau_total*1e9)])
plt.ylabel('tau (ns)')
plt.xlabel('time (ns)')
plt.grid(True, alpha=0.3)
plt.xlim([0,np.max(time0)])
plt.tight_layout()

plt.figure()
plt.plot(R,ne, linestyle='-',linewidth=2)
plt.plot(R_0,ne_2d_0[:, mid], linestyle='--',linewidth=2)
plt.xlabel('R (m)')
plt.ylabel('ne')
plt.legend(['ffw2d','Exp.'])
plt.grid(True, alpha=0.3)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

freq_true = (freq_axis_full-fb)/1e9

ax1.plot(freq_true, fb/1e9, 'b-', linewidth=2, label='fb')
#ax1.set_xlabel('frequency (GHz)')
ax1.set_ylabel('fb (GHz)')
ax1.set_ylim([0,np.max(fb/1e9)])
ax1.plot([fce/1e9,fce/1e9],[0,100],color='black', linestyle='--',linewidth=2, label='fce0')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(freq_true, tau_total*1e9, 'b-', linewidth=2, label='Extracted Group Delay')
ax2.set_xlabel('frequency (GHz)')
ax2.set_ylabel('Delay (ns)')
ax2.set_ylim([0,np.max(tau_total*1e9)])
ax2.plot([fce/1e9,fce/1e9],[0,100],color='black', linestyle='--',linewidth=2, label='fce0')
ax2.grid(True, alpha=0.3)


ax2.legend()
plt.show()
plt.tight_layout()
