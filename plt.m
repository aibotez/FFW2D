clear

R = load('R.dat');
Z = load('Z.dat');
Edis = load('Edis.dat');
Ey_rece = load('receive_E.dat');
ne_2d = load('ne_2d.dat');
rho2d=load('rho_2d.dat');
time = load('time.dat');
ne_base_2d = load('ne_2d_0.dat');

IQ_I = load('IQ_Isig.dat');
IQ_Q = load('IQ_Qsig.dat');
[nant, timel] = size(IQ_I);
idx = find(time*1e9>0);
IQ_I = IQ_I(idx);
IQ_Q = IQ_Q(idx);
time = time(idx);
Ey_rece = Ey_rece(idx);
Metal = load('antenna_geo.dat');


dne = (ne_2d - ne_base_2d)./ne_base_2d;


figure;
subplot(121)
imagesc(R,Z,ne_2d')
colorbar;
subplot(122)
imagesc(R,Z,dne');
colorbar;



[nx, ny] = size(ne_2d);
mid = round(ny/2);
env = abs(hilbert(Edis(:,mid)));
figure;
subplot(2,1,1)
plot(rho2d(:,mid), ne_2d(:,mid), 'LineWidth', 1.5);
subplot(2,1,2)
plot(rho2d(:,mid), env, 'LineWidth', 1.5);


[IQ_I,IQ_Q] = dealIQ(time,IQ_I,IQ_Q);

A_complex = IQ_I + 1i * IQ_Q;
Amplitude_matrix = abs(A_complex);
Phase_matrix     = unwrap(angle(A_complex), [], 2);

V_normalized = A_complex ./ abs(A_complex);
% 重新提取 I 和 Q
I = real(V_normalized);
Q = imag(V_normalized);



nplt = nant+3;
figure;
ax(1) = subplot(nplt,1,1);
plot(time*1e9,Ey_rece);
title('E_y receive')
for i=2:nant+1

	ax(i) = subplot(nplt,1,i);
	plot(time*1e9, IQ_I(i-1,:),time*1e9, IQ_Q(i-1,:),  'LineWidth', 2, 'DisplayName', 'IQ signal');
	title(['P ', num2str(i-1), '：IQ+低通滤波']);
	xlabel('时间 (ns)'); ylabel('信号幅值'); legend; grid on;
end
ax(nplt-1) = subplot(nplt,1,nplt-1);
plot(time*1e9, Phase_matrix, 'LineWidth', 1.5);
title('解调出的等离子体截止层连续相位演化 (Phase Profile)');
xlabel('时间 (ns)'); ylabel('相位 (rad)'); grid on;
subplot(nplt,1,nplt);
for i = 1:nant
	plot(I(i, :), Q(i, :), 'b.');
	axis equal; grid on;
	xlim([-1,1])
	ylim([-1,1])
	hold on
title('I-Q ');
xlabel('I '); ylabel('Q ');
linkaxes(ax,'x')


end


function [I_balanced,Q_balanced] = dealIQ(t,I_raw_matrix,Q_raw_matrix)
	dt = t(2) - t(1);             % 自动计算你的时域仿真步长 (s)
	fs = 1 / dt;                  % 采样频率 (Hz)
	%f0 = 60e9;                    % 托卡马克诊断微波频率 (60 GHz)

	% 设计截止频率：
	% 两倍频在 120 GHz，而边缘台基区的湍流、多普勒频移和杂质响应通常在几 MHz 到几 GHz 级别。
	% 我们把截止频率设为 3 GHz，既能完美斩杀 120 GHz 高频，又能保留丰富的物理细节。
	cutoff_freq = 3.0e9;          

	% 计算归一化截止频率 (MATLAB 要求 Wn 介于 0 和 1 之间，1 代表奈奎斯特频率 fs/2)
	Wn = cutoff_freq / (fs / 2);  

	% 设计一个 4 阶低通巴特沃斯（Butterworth）滤波器，得到分子系数 b 和分母系数 a
	[b, a] = butter(4, Wn, 'low');
	[receive_ants_num,col] = size(I_raw_matrix);

	% --- 2. 预分配滤波后的矩阵空间 ---
	I_filtered = zeros(size(I_raw_matrix));
	Q_filtered = zeros(size(Q_raw_matrix));

	% --- 3. 跨通道批量执行零相位滤波 ---
	for ch = 1:receive_ants_num
		% 提取第 ch 个天线通道的时域信号
		tmp_I = I_raw_matrix(ch, :);
		tmp_Q = Q_raw_matrix(ch, :);
		
		% 🌟 核心核心：使用 filtfilt 进行双向滤波（消除群延迟，确保解调相位绝不滞后）
		I_filtered(ch, :) = filtfilt(b, a, tmp_I);
		Q_filtered(ch, :) = filtfilt(b, a, tmp_Q);

	end

	% --- 4. 自动执行 I/Q 幅度平衡校准（让它们严格关于 0 对称） ---
	I_balanced = I_filtered;
	Q_balanced = Q_filtered;



end
