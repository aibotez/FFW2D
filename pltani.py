import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm  # Progress bar library

# ==========================================
# 1. Load Original Static Data
# ==========================================
Metal = np.loadtxt('antenna_geo.dat')
ne_data0 = np.loadtxt('ne_2d_0.dat')
R = np.loadtxt('R.dat')
Z = np.loadtxt('Z.dat')
time = np.loadtxt('time.dat')
rho2d = np.loadtxt('rho_2d.dat')
output_interval = np.loadtxt('evo.dat')[0]

file_path = "ffw2d_evolution.h5"
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, file_path)
if not os.path.exists(file_path):
    raise FileNotFoundError(f"H5 file not found at:\n{file_path}")

f = h5py.File(file_path, 'r')
print("Datasets in file:", list(f.keys()))

Ez_data = f['Edis']
dn_data = f['ne_2d']

total_frames, nx, ny = Ez_data.shape
print(f"H5 pointers mapped successfully! Total frames: {total_frames}, Grid size: {nx} x {ny}")

# ==========================================
# 2. Downsampling Configuration (Spatial Stride)
# ==========================================
stride = 4

R_sub = R[::stride]
Z_sub = Z[::stride]
ne_data0_sub = ne_data0[::stride, ::stride]
Metal_sub = Metal[::stride, ::stride]
rho2d_sub = rho2d[::stride, ::stride]

extent = [R_sub.min(), R_sub.max(), Z_sub.min(), Z_sub.max()]

frame_0_ez = Ez_data[0, ::stride, ::stride].T
frame_0_dn = dn_data[0, ::stride, ::stride].T
nedata0_T = ne_data0_sub.T
Metal_T = Metal_sub.T
rho2d_T = rho2d_sub.T

dne_0 = np.divide(frame_0_dn - nedata0_T, nedata0_T,
                  out=np.zeros_like(frame_0_dn), where=(nedata0_T > 0.0))

# ==========================================
# 3. Canvas & Layout Initialization (12:5 Rectangle)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
fig.suptitle("FFW2D Evolution", fontsize=16)

# --- Left Plot: Electric Field & Density Background ---
im_ne = ax1.imshow(frame_0_dn, extent=extent, origin='lower', aspect='auto')

colors = [
    (0.3, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.4),
    (1.0, 0.2, 0.2, 0.7),
    (1.0, 1.0, 1.0, 1.0)
]
custom_ramp = LinearSegmentedColormap.from_list('cool_microwave', colors)
E_intensity_0 = np.abs(frame_0_ez)

im_E1 = ax1.imshow(E_intensity_0, extent=extent, origin='lower', aspect='auto', cmap=custom_ramp, vmax=3)

ax1.contour(Metal_T, extent=extent, origin='lower', colors='green', linewidths=0.6, zorder=10)
ax1.contour(rho2d_T, extent=extent, levels=[1.0], origin='lower', colors='black', linewidths=0.6, zorder=10)
ax1.set_title("Electric Field $E_z$ & Density Grid")
ax1.set_xlabel("R (m)")
ax1.set_ylabel("Z (m)")
fig.colorbar(im_E1, ax=ax1, label="Field Intensity")

#cmap='bwr','jet'
# --- Right Plot: Relative Density Perturbation (dne) ---
imdne = ax2.imshow(dne_0, extent=extent, origin='lower', aspect='auto', vmin=np.min(dne_0), vmax=np.max(dne_0))
im_E2 = ax2.imshow(E_intensity_0, extent=extent, origin='lower', aspect='auto', cmap=custom_ramp, vmax=3)

ax2.contour(Metal_T, extent=extent, origin='lower', colors='green', linewidths=0.6, zorder=10)
ax2.contour(rho2d_T, extent=extent, levels=[1.0], origin='lower', colors='black', linewidths=0.6, zorder=10)
ax2.set_title(r"Density Perturbation $\delta n_e$")
ax2.set_xlabel("R (m)")
fig.colorbar(imdne, ax=ax2, label="Density Perturbation Scale")

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='black',
                     bbox=dict(facecolor='white', alpha=0.8), zorder=20)


# ==========================================
# 4. Animation Frame Update Function
# ==========================================


dttimecurent = time[1] - time[0]
dt_per_frame = output_interval * dttimecurent
tot_time = total_frames * dt_per_frame
def update(frame):
    raw_ez_frame = Ez_data[frame, :, :]
    raw_dn_frame = dn_data[frame, :, :]

    current_ez = np.abs(raw_ez_frame[::stride, ::stride].T)
    current_ne = raw_dn_frame[::stride, ::stride].T

    dne = np.divide(current_ne - nedata0_T, nedata0_T, out=np.zeros_like(current_ne), where=(nedata0_T > 0.0))

    im_E1.set_data(current_ez)
    im_E2.set_data(current_ez)
    im_ne.set_data(current_ne)
    imdne.set_data(dne)

    # Calculate real physical time elapsed for current frame
    current_time_val = frame * dt_per_frame
    time_text.set_text(f"Time: {current_time_val:.2f}/{tot_time:.4f} (us)")

    #print(dttimecurent,current_time_val,tot_time)

    return im_ne, im_E1, imdne, im_E2, time_text

# ==========================================
# 5. Execution & Export with Progress Bar
# ==========================================
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=40, blit=False)
# 🌟 1. Initialize the progress bar on terminal
pbar = tqdm(total=total_frames, desc="Rendering Video")
# 🌟 2. Define the callback function that executes every frame
def progress_callback(current_frame, total):
    pbar.update(1)

print("Exporting animation to GIF...")
ani.save('ffw2d_evolution.gif', writer='pillow', fps=25, dpi=100, progress_callback=progress_callback)
pbar.close()

#print("Animation saved successfully as ffw2d_evolution.gif")
plt.show()
plt.close(fig)
f.close()
