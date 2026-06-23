import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# 1. 读取 HDF5 数据
# ==========================================
file_path = "ffw2d_result.h5"  # 你的 H5 文件名
with h5py.File(file_path, 'r') as f:
    # 打印文件内的结构，确保名字对齐
    print("文件内的数据集:", list(f.keys()))

    # 读取静态网格（如果有的话，用于坐标轴）
    # R = f['R_Coordinate'][:]

    # 载入时变三维数据立方体 [Total_Steps, NX, NY]
    # 💡 这里直接切片加载到内存中
    Ez_data = f['Edis'][:]  # 你的电场动态分布

    # 如果你的密度扰动也是时变的，假设叫 'dn_Evolution'：
    dn_data = f['ne_2d'][:]
ne_data = np.loadtxt('ne_2d_0.dat')
R = np.loadtxt('R.dat')
Z = np.loadtxt('Z.dat')
extent=[R.min(), R.max(), Z.min(), Z.max()]
total_frames, nx, ny = Ez_data.shape
print(f"成功载入数据！总帧数: {total_frames}, 网格大小: {nx} x {ny}")

# ==========================================
# 2. 初始化画布与子图布局
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Tokamak Simulation Dynamic Evolution", fontsize=16)

# ─── 左图：电场 Ez 动态分布 ───
# 物理仿真中电场常有正负，用 'seismic' 或 'RdBu' 渐变色最为直观
vmax_ez = np.max(np.abs(Ez_data)) * 0.8  # 稍微缩放色彩范围让波纹更明显
im1 = ax1.imshow(Ez_data[0].T,extent=extent, cmap='jet', 
                 vmin=-5, vmax=5, origin='lower')
ax1.set_title("Electric Field $E_z$")
ax1.set_xlabel("Y (Grid)")
ax1.set_ylabel("X (Grid)")
fig.colorbar(im1, ax=ax1, label="Field Intensity")

# ─── 右图：密度扰动 ───
vmax_dn = np.max(np.abs(dn_data)) * 0.8
im2 = ax2.imshow(dn_data[0].T,extent=extent, cmap='viridis',
                 vmin=-vmax_dn, vmax=vmax_dn, origin='lower')
ax2.set_title(r"Density Perturbation $\delta n_e$")
ax2.set_xlabel("Y (Grid)")
fig.colorbar(im2, ax=ax2, label="Perturbation Amplitude")

# 时间文本标签
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='black',
                     bbox=dict(facecolor='white', alpha=0.8))


# ==========================================
# 3. 🌟 核心：动画刷新函数（高效更新数据，不重建图像）
# ==========================================
def update(frame):
    # 使用 set_data 机制可以避免重复创建 ax.imshow，极大地提升渲染流畅度
    im1.set_data(Ez_data[frame].T)
    im2.set_data(dn_data[frame].T)

    # 更新帧数文本
    time_text.set_text(f"Frame: {frame}/{total_frames - 1}")

    # 返回需要被重新绘制的图表对象
    return im1, im2, time_text


# ==========================================
# 4. 启动与保存动画
# ==========================================
# interval=100 表示每帧间隔 100 毫秒
ani = animation.FuncAnimation(fig, update, frames=total_frames,
                              interval=100, blit=True)

# 或者是直接弹窗看动态交互式动画
plt.tight_layout()
plt.show()

# 💡 如果想保存为高质量视频（需要系统安装了 ffmpeg）：
# print("正在导出视频为 simulation_evolution.mp4...")
# ani.save('simulation_evolution.mp4', writer='ffmpeg', fps=10, dpi=150)
# print("导出成功！")