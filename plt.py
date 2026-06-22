import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert

# Configure matplotlib to ensure proper font rendering for labels
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. Load Data and Preprocessing
# ==========================================
R = np.loadtxt('R.dat')
Z = np.loadtxt('Z.dat')
Edis = np.loadtxt('Edis.dat')
Ey_rece = np.loadtxt('receive_E.dat')
ne_2d = np.loadtxt('ne_2d.dat')
rho2d = np.loadtxt('rho_2d.dat')
time = np.loadtxt('time.dat')
ne_base_2d = np.loadtxt('ne_2d_0.dat')

IQ_I_raw = np.loadtxt('IQ_Isig.dat')
IQ_Q_raw = np.loadtxt('IQ_Qsig.dat')
Metal = np.loadtxt('antenna_geo.dat')

# Handle dimensions if IQ data is loaded as a 1D array
if IQ_I_raw.ndim == 1:
    IQ_I_raw = IQ_I_raw.reshape(1, -1)
    IQ_Q_raw = IQ_Q_raw.reshape(1, -1)

nant, timel = IQ_I_raw.shape

# Filter data to include only time > 0
idx = np.where(time * 1e9 > 0)[0]
time = time[idx]
Ey_rece = Ey_rece[idx]
IQ_I_raw = IQ_I_raw[:, idx]
IQ_Q_raw = IQ_Q_raw[:, idx]
nx, ny = ne_2d.shape
dne = np.zeros((nx,ny))
# Calculate relative density fluctuation (dne)
dne = np.divide(ne_2d - ne_base_2d, ne_base_2d, out=np.zeros_like(ne_2d), where=(ne_base_2d > 0.0))

# ==========================================
# 2. IQ Filter Function Definition
# ==========================================
def dealIQ(t, I_raw_matrix, Q_raw_matrix):
    dt = t[1] - t[0]         # Automatic calculation of time step (s)
    fs = 1 / dt              # Sampling frequency (Hz)
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

# Execute filtering and balance calibration
IQ_I, IQ_Q = dealIQ(time, IQ_I_raw, IQ_Q_raw)

# ==========================================
# 3. Signal Demodulation: Amplitude and Phase
# ==========================================
A_complex = IQ_I + 1j * IQ_Q
Amplitude_matrix = np.abs(A_complex)
# Unwrap phase along the time axis (equivalent to MATLAB dim=2)
Phase_matrix = np.unwrap(np.angle(A_complex), axis=-1)

V_normalized = A_complex / np.abs(A_complex)
I_norm = np.real(V_normalized)
Q_norm = np.imag(V_normalized)

# ==========================================
# Plot 1: 2D Electron Density and Fluctuations
# ==========================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
im1 = plt.imshow(ne_2d.T, extent=[R.min(), R.max(), Z.min(), Z.max()], origin='lower', aspect='auto', cmap='jet')
plt.title('2D Electron Density Profile (ne)')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.colorbar(im1)

plt.subplot(1, 3, 2)
im2 = plt.imshow(dne.T, extent=[R.min(), R.max(), Z.min(), Z.max()], origin='lower', aspect='auto', cmap='bwr', vmin=np.min(dne), vmax=np.max(dne))
#im2 = plt.imshow(dne.T, extent=[R.min(), R.max(), Z.min(), Z.max()], origin='lower', aspect='auto', cmap='bwr')
plt.title('Relative Density Fluctuation (dne/ne0)')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.colorbar(im2)
plt.subplot(1, 3, 3)
im3 = plt.imshow(Edis.T, extent=[R.min(), R.max(), Z.min(), Z.max()], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(im3)
plt.tight_layout()

# ==========================================
# Plot 2: Midplane Profiles and Field Envelope
# ==========================================

mid = int(ny / 2)
# Compute the 1D Hilbert envelope of the electric field at the midplane
env = np.abs(hilbert(Edis[:, mid]))

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(rho2d[:, mid], ne_2d[:, mid], 'b-', label='ne')
plt.ylabel('ne (m^-3)')
plt.grid(True)
plt.title('Midplane Radial Profiles')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(rho2d[:, mid], env, 'r-', label='E-field Envelope')
plt.xlabel('rho (Normalized Radius)')
plt.ylabel('E_dis Envelope (a.u.)')
plt.grid(True)
plt.legend()
plt.tight_layout()

# ==========================================
# Plot 3: Time-domain Signals and Constellation
# ==========================================
nplt = nant + 3
fig, axes = plt.subplots(nplt, 1, figsize=(10, 2.5 * nplt), sharex=False)

# 1. Received raw Ey field
axes[0].plot(time * 1e9, Ey_rece, 'k-')
axes[0].set_title('Raw Ey Received Signal')
axes[0].set_ylabel('Amplitude')
axes[0].grid(True)

# 2. Demodulated I and Q channels for each antenna
for i in range(1, nant + 1):
    axes[i].plot(time * 1e9, IQ_I[i-1, :], 'r-', linewidth=1.5, label='I (In-phase)')
    axes[i].plot(time * 1e9, IQ_Q[i-1, :], 'g-', linewidth=1.5, label='Q (Quadrature)')
    axes[i].set_title(f'Channel P {i}: Filtered IQ Signals')
    axes[i].set_ylabel('Amplitude')
    axes[i].legend()
    axes[i].grid(True)

# 3. Continuous phase evolution profile
axes[nplt-2].plot(time * 1e9, Phase_matrix.T, linewidth=1.5)
axes[nplt-2].set_title('Unwrapped Phase Tracking (Plasma Cut-off Layer)')
axes[nplt-2].set_xlabel('Time (ns)')
axes[nplt-2].set_ylabel('Phase (rad)')
axes[nplt-2].grid(True)

# Link time axes for zoom/pan synchronization across time-domain plots
for ax in axes[:-1]:
    if ax != axes[0]:
        ax.sharex(axes[0])

# 4. Normalized I-Q vector constellation plot
ax_iq = axes[nplt-1]
for i in range(nant):
    ax_iq.plot(I_norm[i, :], Q_norm[i, :], '.', markersize=2, label=f'Ch {i+1}')
ax_iq.set_aspect('equal')
ax_iq.set_xlim([-1.1, 1.1])
ax_iq.set_ylim([-1.1, 1.1])
ax_iq.set_title('Normalized I-Q Constellation Diagram')
ax_iq.set_xlabel('I')
ax_iq.set_ylabel('Q')
ax_iq.grid(True)
plt.tight_layout()

# ==========================================
# Plot 4: Complex Signal FFT Spectral Analysis
# ==========================================
plt.figure(figsize=(10, 5))

dt = time[1] - time[0]
fs = 1 / dt
N = len(time)

# Calculate both positive and negative frequencies for Doppler shifts
freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dt))

for ch in range(nant):
    # Extract the full complex signal: A(t) = I(t) + i*Q(t)
    signal_complex = A_complex[ch, :]
    
    # Remove DC offset to eliminate the central 0 Hz spike
    signal_complex = signal_complex - np.mean(signal_complex)
    
    # Execute Fast Fourier Transform and center the spectrum
    fft_vals = np.fft.fftshift(np.fft.fft(signal_complex))
    
    # Calculate Power Spectral Density (PSD) in dB
    psd = 20 * np.log10(np.abs(fft_vals) / N + 1e-12)
    
    # Convert frequency axis to MHz for tokamak diagnostic visualization
    plt.plot(freqs / 1e6, psd, linewidth=1.5, label=f'Channel {ch+1}')

plt.title('Power Spectral Density of Complex IQ Signal (Doppler Frequency Shift)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power Spectrum Magnitude (dB)')
plt.grid(True)
# Restrict display window to focus on turbulent and Doppler features
plt.xlim([-500, 500]) 
plt.legend()
plt.tight_layout()

# Render all generated windows
plt.show()