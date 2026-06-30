import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from matplotlib.colors import LinearSegmentedColormap



def get_fft(time, nant, A_complex, freq_span):
    dt = time[1] - time[0]
    fs = 1 / dt
    lag = 10
    A_complex = np.tile(A_complex, (1, lag))
    
    print(len(time))
    N_fft = 512
    
    freqs = np.fft.fftshift(np.fft.fftfreq(N_fft, d=dt))
    freq = freqs
    
    idx = np.where((freq < freq_span[1]) & (freq > freq_span[0]))[0]
    freq_filtered = freqs[idx]
    psd = np.zeros((nant, len(idx)))
    
    for ch in range(nant):
        signal_complex = A_complex[ch, :]
        signal_complex = signal_complex - np.mean(signal_complex)
        window = np.blackman(len(signal_complex))
        signal_complex = signal_complex * window
        fft_vals = np.fft.fftshift(np.fft.fft(signal_complex, n=N_fft))
        amplitude = np.abs(fft_vals) / len(time)
        psd0 = 20 * np.log10(amplitude + 1e-6)
        
        psd[ch, :] = psd0[idx]
        
    return freq_filtered, psd


def dealIQ(t, I_raw_matrix, Q_raw_matrix):
    dt = t[1] - t[0]         # Automatic calculation of time step (s)
    fs = 1 / dt             # Sampling frequency (Hz)
    cutoff_freq = 3.0e9      # Cutoff frequency set to 3 GHz
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


Basic_par = np.loadtxt("Basic_par.dat")

nant = int(Basic_par[3])
slices_tot=int(Basic_par[1])
time = np.loadtxt("time_slice.dat")
time_s0 = np.loadtxt("time.dat")
lent = len(time_s0)

IQI = np.zeros((nant,slices_tot))
IQQ = np.zeros((nant,slices_tot))
AIQ = np.zeros((nant,slices_tot))
Phase_matrixs = np.zeros((nant,slices_tot))
A_complexs = np.zeros((nant,slices_tot))
for i in range(slices_tot):
    IQ_Ifile = 'IQ_Isig_{}.dat'.format(str(i))
    IQ_Qfile = 'IQ_Qsig_{}.dat'.format(str(i))
    IQ_I_raw = np.loadtxt(IQ_Ifile)
    IQ_Q_raw = np.loadtxt(IQ_Qfile)
    
    IQ_I_raw = IQ_I_raw[:,int(lent/2):lent]
    IQ_Q_raw = IQ_Q_raw[:,int(lent/2):lent]
    time_1s = time_s0[int(lent/2):lent]
    #time_1s = [i*dtfd for i in range(int(lent/2))]
    IQ_I, IQ_Q = dealIQ(time_1s/1e9, IQ_I_raw, IQ_Q_raw)
    IQI[:,i] = IQ_I[:,-1]
    IQQ[:,i] = IQ_Q[:,-1]
    A_complex = IQ_I + 1j * IQ_Q
    Phase_matrix = np.angle(A_complex)
    Amplitude_matrix = np.abs(A_complex)
    AIQ[:,i] = Amplitude_matrix[:,-1]
    A_complexs[:,i] = A_complex[:,-1]
    Phase_matrixs[:,i] = Phase_matrix[:,-1]


Phase_matrixs = np.array(Phase_matrixs)
Phase_matrixs = np.unwrap(Phase_matrixs)

plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
plt.figure(figsize=(12, 8))
plt.subplot(221)
#plt.plot(np.array(time_1s),Phase_matrix.T)
plt.plot(time*1000,Phase_matrixs.T)
plt.legend(['P1','P2'])
plt.xlabel('time (ms)')
plt.ylabel('phase (rad)')

plt.grid(True)
plt.subplot(222)
plt.plot(time*1000,AIQ.T)
plt.xlabel('time (ms)')
plt.ylabel('IQ amplitude')
plt.legend(['P1','P2'])
plt.grid(True)
plt.subplot(223)
[freq,psd] = get_fft(time,nant,A_complexs,[0,1e9])
plt.plot(freq/1000,psd.T)
plt.xlabel('f (kHz)')
plt.ylabel('IQ complexs fft')
plt.legend(['P1','P2'])
plt.grid(True)
plt.subplot(224)
[freq,psd] = get_fft(time,nant,AIQ,[0,1e9])
plt.plot(freq/1000,psd.T)
plt.xlabel('f (kHz)')
plt.ylabel('IQ amplitude fft')
plt.legend(['P1','P2'])
plt.grid(True)
plt.tight_layout()
plt.show()





