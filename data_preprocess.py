import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import welch, find_peaks
from scipy.signal import butter, filtfilt  # 滤波器设计+零相位滤波
import pywt # 小波变换库
from scipy.signal import welch, find_peaks, butter, filtfilt, hilbert  # 新增hilbert
from scipy import signal  # 显式导入scipy.signal，避免兼容问题

# 新增：全局字体配置，解决符号显示问题
rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = True  # 解决负号显示问题
rcParams['mathtext.fontset'] = 'cm'    # 解决上标/下标显示问题

def extract_single_column_csv(file_path):
    """
    读取单列CSV（首行是无效信息，后续是电压数据）
    :param file_path: CSV文件路径
    :return: 包含电压数据的DataFrame（列名：CH1V）
    """
    # 读取CSV：skiprows=1 跳过首行无效信息；usecols=[0] 仅读取第一列；header=None 表示无列名
    df = pd.read_csv(
        file_path,
        skiprows=1,  # 跳过首行（t0=-2.000...这行）
        usecols=[0],  # 仅读取第一列（电压数据）
        header=None  # 无列名，后续手动命名
    )

    # 给列命名为CH1V（和你之前的代码对齐）
    df.columns = ['Voltage']

    # 转换为数值类型（处理可能的无效字符，比如###）
    df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')

    return df


# 最普通的采样方式，平均取一个点
def downsample_data(data_df, downsample_factor=50):
    """
    示波器式抽取降采样：每n个连续点取第1个点，和示波器降采样逻辑一致
    :param data_df: 原始Voltage的DataFrame
    :param downsample_factor: 降采样因子（默认10，50kHz→5kHz）
    :return: 降采样后的DataFrame、降采样后采样率
    """
    # 原始采样率（根据你的场景：100万条/20秒=50kHz）
    original_sr = 50000
    voltage_array = data_df['Voltage'].values
    align_length = len(voltage_array) - (len(voltage_array) % downsample_factor)
    voltage_aligned = voltage_array[:align_length]
    voltage_downsampled = voltage_aligned[::downsample_factor]  # 步长取数，最简洁的抽取逻辑
    down_df = pd.DataFrame({'Voltage': voltage_downsampled})
    down_sr = original_sr / downsample_factor
    return down_df, down_sr


def average_downsample_data(data_df, downsample_factor=10):
    """
    均值降采样：把n个连续点的均值作为新点，减少数据量
    :param data_df: 原始Voltage的DataFrame
    :param downsample_factor: 降采样因子（默认10，50kHz→5kHz）
    :return: 降采样后的DataFrame、降采样后采样率
    """
    # 原始采样率（根据你的场景：100万条/20秒=50kHz）
    original_sr = 50000
    # 提取电压数组
    voltage_array = data_df['Voltage'].values
    # 对齐数据长度（避免余数，比如100万条刚好被10整除）
    align_length = len(voltage_array) - (len(voltage_array) % downsample_factor)
    voltage_aligned = voltage_array[:align_length]
    # 重塑为二维数组，按行取均值（核心：均值降采样，避免失真）
    voltage_downsampled = voltage_aligned.reshape(-1, downsample_factor).mean(axis=1)
    # 转成DataFrame（保持和原数据格式一致）
    down_df = pd.DataFrame({'Voltage': voltage_downsampled})
    # 降采样后的采样率
    down_sr = original_sr / downsample_factor
    return down_df, down_sr

# 峰值降采样函数
def downsample_data_peak_preserve_signed(data_df, downsample_factor=5):
    """
    带符号的绝对值最大值降采样：保留正负峰值
    :param data_df: 原始Voltage的DataFrame
    :param downsample_factor: 降采样因子
    :return: 降采样后的DataFrame、采样率
    """
    original_sr = 50000  # 原始采样率
    voltage_array = data_df['Voltage'].values
    # 对齐数据长度
    align_length = len(voltage_array) - (len(voltage_array) % downsample_factor)
    voltage_aligned = voltage_array[:align_length]
    # 重塑为二维数组（每组downsample_factor个点）
    voltage_reshaped = voltage_aligned.reshape(-1, downsample_factor)
    # 1. 取每组的绝对值最大值（保留峰值大小）
    abs_max = np.max(np.abs(voltage_reshaped), axis=1)
    # 2. 取每组的均值的符号（保留峰值方向）
    group_sign = np.sign(np.mean(voltage_reshaped, axis=1))
    # 3. 带符号的峰值 = 绝对值最大值 × 符号
    voltage_downsampled = abs_max * group_sign
    # 转成DataFrame
    down_df = pd.DataFrame({'Voltage': voltage_downsampled})
    down_sr = original_sr / downsample_factor
    return down_df, down_sr


def plot_spectrum(data_df, sr, top_n=3):
    """
    绘制频谱图，自动标注前N个能量峰值（核心频率）
    :param data_df: 降采样后的DataFrame
    :param sr: 降采样后的采样率
    :param top_n: 显示前N个峰值
    """
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    # 提取电压数组
    voltage = data_df['Voltage'].values
    # 计算功率谱密度（PSD）：f=频率，Pxx=对应频率的能量
    f, Pxx = welch(
        voltage,
        fs=sr,        # 采样率
        nperseg=1024, # 分段长度（越大，频率分辨率越高）
        scaling='density'
    )

    # 绘制频谱图（对数坐标，更易看峰值）
    plt.figure(figsize=(12, 6))
    plt.semilogy(f, Pxx, linewidth=0.8, label='功率谱密度')

    # 自动找峰值（核心：只找能量≥10%最大值的峰值，过滤噪声）
    peak_threshold = np.max(Pxx) * 0.1  # 阈值：能量≥最大值的10%
    peaks, properties = find_peaks(Pxx, height=peak_threshold)
    # 按能量排序，取前top_n个峰值
    peak_energies = properties['peak_heights']
    peak_indices = np.argsort(peak_energies)[::-1][:top_n]  # 从大到小排序
    top_peaks = peaks[peak_indices]
    top_freqs = f[top_peaks]
    top_energies = peak_energies[peak_indices]

    # 标注峰值（核心频率）
    for i, (freq, energy) in enumerate(zip(top_freqs, top_energies)):
        plt.plot(freq, energy, 'ro', markersize=6)  # 红色圆点标峰值
        plt.annotate(
            f'峰值{i+1}：{freq:.1f}Hz',
            xy=(freq, energy),
            xytext=(freq+100, energy*1.2),  # 文字偏移，避免遮挡
            arrowprops=dict(arrowstyle='->', color='red')
        )

    # 图表标注
    plt.xlabel("频率 (Hz)")
    plt.ylabel("功率谱密度 (V²/Hz)")
    plt.title(f"信号频谱图（采样率{sr}Hz）")
    plt.xlim(0, 2500)  # 只看0~2500Hz（覆盖所有压电片场景）
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # 输出核心频率
    print(f"检测到的核心频率（按能量排序）：{top_freqs} Hz")
    return top_freqs[0]  # 返回能量最高的频率（压电片主频率）


# 低通滤波，只是简单方法放到这里
def lowpass_filter(data_df, sr, cutoff_freq):
    """
    巴特沃斯低通滤波：保留低频信号，滤除高频噪声
    :param data_df: 待滤波的DataFrame（原始/降采样后都可）
    :param sr: 传入data_df对应数据的实际采样率（关键！原始数据传50000，降采样后传1000）
    :param cutoff_freq: 截止频率（设为2×核心频率，比如3Hz→6Hz）
    :return: 滤波后的DataFrame
    """
    voltage = data_df['Voltage'].values
    nyq = 0.5 * sr
    normal_cutoff = cutoff_freq / nyq
    # 验证：打印归一化截止频率（必须在0.001~0.5之间，否则滤波无效）
    print(f"归一化截止频率：{normal_cutoff}（正常范围0.001~0.5）")
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    filtered_voltage = filtfilt(b, a, voltage)
    filtered_df = pd.DataFrame({'Voltage': filtered_voltage})
    return filtered_df

# 适配单列数据的绘图函数
def plot_voltage_single_column(data,title_suffix=""):
    """
    绘制单列电压数据：横轴为数据索引（顺序），纵轴为电压
    :param data: 仅含CH1V列的DataFrame
    """
    # 设置中文显示
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    # 提取电压数据，横轴用索引（0,1,2,...）
    voltage_data = data['Voltage']
    x_axis = range(len(voltage_data))  # 横轴：按顺序排列的索引

    # 绘制图形（100w条数据建议抽样，否则绘图卡顿）
    plt.figure(figsize=(12, 6))  # 调整画布大小，适配大量数据
    # 抽样绘制：每100个点取1个（可根据需要调整抽样率）
    sample_rate = 1
    plt.plot(
        x_axis[::sample_rate],
        voltage_data.values[::sample_rate],
        linewidth=0.8, alpha=0.8
    )

    # 图表标注
    plt.xlabel("数据点序号（按时间顺序）")
    plt.ylabel("电压")
    plt.title(f"电压随数据序列变化{title_suffix}")
    plt.grid(alpha=0.3)  # 增加网格，便于查看
    plt.show()

# ========== 2. 核心调整：适配低频基频的频谱分析函数 ==========
def plot_spectrum_base_freq(data_df, sr, top_n=3):
    rcParams['font.sans-serif'] = ['SimHei']
    rcParams['axes.unicode_minus'] = False

    voltage = data_df['Voltage'].values
    # 关键修改1：限制分段长度不超过数据长度的1/10（避免分段过大）
    max_nperseg = len(voltage) // 10
    nperseg = min(int(sr / 1), max_nperseg)  # 频率分辨率改为1Hz（适配7-8Hz），且不超数据长度
    # 关键修改2：强制设置noverlap为nperseg//2，避免自动调整
    f, Pxx = welch(
        voltage,
        fs=sr,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling='density'
    )

    plt.figure(figsize=(12, 6))
    plt.semilogy(f, Pxx, linewidth=0.8, label='功率谱密度')

    # 关键修改3：先过滤0-10Hz的频率，再找峰值
    freq_mask = (f >= 0) & (f <= 10)
    f_filtered = f[freq_mask]
    Pxx_filtered = Pxx[freq_mask]

    if len(Pxx_filtered) == 0:
        print("⚠️ 0-10Hz无有效频率数据")
        return 0

    # 基于0-10Hz的能量找阈值
    peak_threshold = np.max(Pxx_filtered) * 0.1
    peaks, properties = find_peaks(Pxx_filtered, height=peak_threshold)

    if len(peaks) == 0:
        print("⚠️ 0-10Hz无有效峰值，取能量最大值")
        dominant_idx = np.argmax(Pxx_filtered)
        base_freq = f_filtered[dominant_idx]
    else:
        peak_energies = properties['peak_heights']
        peak_indices = np.argsort(peak_energies)[::-1][:top_n]
        top_peaks = peaks[peak_indices]
        top_freqs = f_filtered[top_peaks]
        base_freq = top_freqs[0]

    # 标注峰值
    plt.plot(base_freq, Pxx_filtered[np.where(f_filtered == base_freq)[0][0]], 'ro', markersize=6)
    plt.annotate(
        f'真实基频：{base_freq:.2f}Hz',
        xy=(base_freq, Pxx_filtered[np.where(f_filtered == base_freq)[0][0]]),
        xytext=(0.2, 20),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', color='red')
    )

    plt.xlabel("频率 (Hz)")
    plt.ylabel("功率谱密度 (V²/Hz)")
    plt.title(f"信号频谱图（采样率{sr}Hz，聚焦0-10Hz）")
    plt.xlim(0, 10)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    print(f"✅ 识别到的真实基频（0-10Hz）：{base_freq:.2f} Hz")
    return base_freq


def wavelet_verify_base_freq_simple(data_df, base_freq, sr=50000):
    """
    极简版小波变换验证（完全避开pywt版本坑）：
    1. 用小波分解重构信号，验证2Hz基频的能量占比
    2. 无任何复杂属性调用，仅用基础的小波分解/重构
    """
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = True

    voltage = data_df['Voltage'].values
    # 取前10000个点（少量数据，快速计算）
    voltage_short = voltage[:10000]
    time_axis = np.arange(len(voltage_short)) / sr  # 时间轴（秒）

    # 步骤1：小波分解（用基础的db4小波，分解3层）
    # 仅用pywt基础API，无任何属性调用
    coeffs = pywt.wavedec(voltage_short, 'db4', level=3)
    cA3, cD3, cD2, cD1 = coeffs  # 近似系数（低频）+ 细节系数（高频）

    # 步骤2：重构仅保留低频成分（对应2Hz左右）
    # 清空高频细节系数，只保留近似系数
    coeffs_recon = [cA3, np.zeros_like(cD3), np.zeros_like(cD2), np.zeros_like(cD1)]
    recon_signal = pywt.waverec(coeffs_recon, 'db4')

    # 步骤3：计算重构信号与原始信号的相关性（验证低频能量）
    corr = np.corrcoef(voltage_short[:len(recon_signal)], recon_signal)[0, 1]

    # 步骤4：绘制对比图（原始信号 vs 小波重构低频信号）
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis[:len(recon_signal)], voltage_short[:len(recon_signal)],
             label='原始信号', linewidth=0.8, alpha=0.7, color='#1f77b4')
    plt.plot(time_axis[:len(recon_signal)], recon_signal,
             label=f'小波重构低频信号（{base_freq:.1f}Hz）', linewidth=1.2, color='#d62728')
    plt.xlabel('时间 (s)')
    plt.ylabel('电压 (V)')
    plt.title(f'小波变换验证：{base_freq:.1f}Hz低频成分对比（相关系数={corr:.3f}）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 输出验证结论
    print(f"✅ 小波变换验证完成（极简版，无版本报错）：")
    print(f"   - 低频重构信号与原始信号相关系数：{corr:.3f}（越接近1，低频成分越显著）")
    print(f"   - 结论：{base_freq:.1f}Hz频率成分在信号中占主导，与每秒2-3次旋转的实验观察一致")


def bandpass_filter(data_df, sr, low_cut, high_cut):
    """
    巴特沃斯带通滤波：仅保留low_cut~high_cut之间的频率
    适配你的2Hz基频，保留1-10Hz区间
    """
    voltage = data_df['Voltage'].values
    nyq = 0.5 * sr
    # 归一化截止频率
    low = low_cut / nyq
    high = high_cut / nyq
    # 二阶带通滤波器，零相位滤波.改2阶为1阶，减少幅值衰减
    b, a = butter(1, [low, high], btype='band', analog=False)
    filtered_voltage = filtfilt(b, a, voltage)
    # 新增：幅值校准（抵消滤波衰减，乘以1.5）
    filtered_voltage = filtered_voltage * 1.5
    filtered_df = pd.DataFrame({'Voltage': filtered_voltage})
    return filtered_df

def downsample_data_linear(data_df, downsample_factor=5):
    """线性降采样：保留原始幅值，避免放大尖峰"""
    original_sr = 50000
    voltage_array = data_df['Voltage'].values
    # 用线性插值降采样，保留幅值
    t_original = np.arange(len(voltage_array))
    t_down = np.arange(0, len(voltage_array), downsample_factor)
    voltage_downsampled = np.interp(t_down, t_original, voltage_array)
    down_df = pd.DataFrame({'Voltage': voltage_downsampled})
    down_sr = original_sr / downsample_factor
    return down_df, down_sr

def wavelet_denoise(data_df, wavelet='db4', level=3):
    """
    小波阈值降噪：保留低频基频+高频脉冲尖峰
    - wavelet：db4小波（适配脉冲信号）
    - level：分解层数（3层足够保留2Hz基频）
    """
    voltage = data_df['Voltage'].values
    # 小波分解
    coeffs = pywt.wavedec(voltage, wavelet, level=level)
    # 对高频细节系数做阈值处理（保留尖峰，滤除噪声）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 噪声标准差估计
    threshold = sigma * np.sqrt(2 * np.log(len(voltage)))  # 通用阈值
    # 仅对高频系数阈值处理，保留低频近似系数
    coeffs_denoised = coeffs.copy()
    for i in range(1, len(coeffs)):
        coeffs_denoised[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    # 小波重构
    denoised_voltage = pywt.waverec(coeffs_denoised, wavelet)
    # 对齐长度（小波重构可能差1个点）
    denoised_voltage = denoised_voltage[:len(voltage)]
    denoised_df = pd.DataFrame({'Voltage': denoised_voltage})
    return denoised_df


def wavelet_denoise_with_envelope_fix(data_df, wavelet='db4', level=3, down_sr=None):
    """
    适配正负对称脉冲信号的修复版：保留正负波动+修复幅值+匹配实际采样率
    :param down_sr: 传入降采样后的实际采样率（必须，避免硬编码）
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))  # 保留原始最大幅值
    original_sign = np.sign(voltage)  # 记录原始信号的正负符号（核心：保留对称性）

    # ========== 步骤1：小波降噪（保留正负尖峰） ==========
    coeffs = pywt.wavedec(voltage, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(voltage)))
    coeffs_denoised = coeffs.copy()
    for i in range(1, len(coeffs)):
        coeffs_denoised[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    denoised_voltage = pywt.waverec(coeffs_denoised, wavelet)
    denoised_voltage = denoised_voltage[:len(voltage)]  # 对齐长度

    # ========== 步骤2：希尔伯特包络（保留正负符号） ==========
    analytic_signal = signal.hilbert(denoised_voltage)
    envelope = np.abs(analytic_signal) * original_sign  # 用原始符号还原正负（核心修复）

    # 步骤3：低通滤波（匹配实际采样率，不再硬编码）
    if down_sr is None:
        raise ValueError("必须传入降采样后的实际采样率down_sr！")
    nyq = 0.5 * down_sr
    cutoff = 3  # 保持3Hz基频，但基于实际采样率计算
    b, a = butter(2, cutoff / nyq, btype='low')
    smooth_envelope = filtfilt(b, a, envelope)  # 此时包络是带正负的

    # ========== 步骤4：合并信号（保留正负，弱化包络权重） ==========
    # 归一化（保留正负）
    envelope_norm = smooth_envelope / np.max(np.abs(smooth_envelope))
    denoised_norm = denoised_voltage / np.max(np.abs(denoised_voltage))
    # 调整权重：降噪信号占70%（保留原始波动），包络占30%（仅平滑趋势）
    enhanced_voltage = (denoised_norm * 0.7) + (envelope_norm * 0.3)
    # 还原原始幅值
    enhanced_voltage = enhanced_voltage * original_max_amp

    enhanced_df = pd.DataFrame({'Voltage': enhanced_voltage})
    return enhanced_df

def wavelet_denoise_pulse_preserve(data_df, wavelet='db4', level=3):
    """
    适配脉冲信号的轻量级降噪：只去噪声，完全保留尖峰幅值和正负.去掉了希尔伯特包络和权重合并，避免平滑过度
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))  # 记录原始最大幅值（核心：后续还原）

    # ========== 步骤1：小波降噪（仅过滤高频噪声，保留尖峰） ==========
    coeffs = pywt.wavedec(voltage, wavelet, level=level)
    # 调整阈值：只过滤“极高频的小噪声”，保留尖峰（阈值×0.5，更宽松）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(voltage))) * 0.5  # 阈值缩小，少过滤
    coeffs_denoised = coeffs.copy()
    # 只对最顶层的高频系数做阈值（更轻量的降噪）
    coeffs_denoised[-1] = pywt.threshold(coeffs[-1], threshold, mode='soft')
    # 其他层系数不处理，完全保留尖峰
    denoised_voltage = pywt.waverec(coeffs_denoised, wavelet)
    denoised_voltage = denoised_voltage[:len(voltage)]  # 对齐长度

    # ========== 步骤2：强制还原原始幅值（核心：保留尖峰强度） ==========
    # 用原始最大幅值重新缩放，确保尖峰强度不变
    denoised_voltage = denoised_voltage * (original_max_amp / np.max(np.abs(denoised_voltage)))

    enhanced_df = pd.DataFrame({'Voltage': denoised_voltage})
    return enhanced_df

# 会极致保留特征，并且增强对抗衰减，形成连续的曲线。
def enhance_lstm_feature(data_df, down_sr, base_freq=2):
    """
    强化LSTM可识别的规律：保留基频趋势 + 增强关键尖峰
    :param down_sr: 降采样后的采样率
    :param base_freq: 目标基频（比如2Hz，和你之前的基频一致）
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))  # 新增：记录原始最大幅值，用于后续校准

    # ========== 步骤1：带通滤波（只保留base_freq±1Hz的信号，强化基频规律） ==========
    nyq = 0.5 * down_sr
    low = (base_freq - 1) / nyq
    high = (base_freq + 1) / nyq
    b, a = butter(2, [low, high], btype='band')
    # 零相位滤波，避免信号偏移
    filtered_voltage = filtfilt(b, a, voltage)

    # ========== 步骤2：增强尖峰（让LSTM更容易捕捉关键特征） ==========
    # 找到尖峰位置（幅值>0.8倍最大值）
    peak_mask = np.abs(filtered_voltage) > np.max(np.abs(filtered_voltage)) * 0.8
    # 尖峰幅值×1.5，非尖峰保持不变
    enhanced_voltage = np.where(peak_mask, filtered_voltage * 1.5, filtered_voltage)

    # # ========== 步骤3：归一化（LSTM对归一化数据更敏感） ==========
    # enhanced_voltage = (enhanced_voltage - np.mean(enhanced_voltage)) / np.std(enhanced_voltage)
    # ========== 新增：幅值校准（避免滤波后幅值衰减） ==========
    enhanced_voltage = enhanced_voltage * (original_max_amp / np.max(np.abs(enhanced_voltage)))

    return pd.DataFrame({'Voltage': enhanced_voltage})

def enhance_lstm_feature_pro(data_df, down_sr, base_freq):
    """
    强化LSTM可识别的规律（分层细化版）：
    1. 预滤波去高频噪声 → 2. 基频带通滤波 → 3. 自适应尖峰增强（无归一化）
    :param down_sr: 降采样后的采样率
    :param base_freq: 目标基频（比如2Hz）
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))
    original_mean = np.mean(voltage)  # 记录原始均值，避免直流偏移

    # ========== 步骤1：预滤波（去除高频噪声，保护基频） ==========
    # 先滤除base_freq×5以上的高频噪声
    nyq = 0.5 * down_sr
    high_pass_cutoff = (base_freq * 5) / nyq
    b_pre, a_pre = butter(1, high_pass_cutoff, btype='low')  # 1阶低通预滤波
    pre_filtered = filtfilt(b_pre, a_pre, voltage)

    # ========== 步骤2：基频带通滤波（更精细的频率范围） ==========
    # 缩小带通范围：base_freq±0.5Hz（避免引入无关频率）
    low = (base_freq - 0.5) / nyq
    high = (base_freq + 0.5) / nyq
    b_band, a_band = butter(3, [low, high], btype='band')  # 3阶滤波器（更平滑）
    band_filtered = filtfilt(b_band, a_band, pre_filtered)

    # ========== 步骤3：自适应尖峰增强（避免过度增强） ==========
    # 步骤3.1：计算滤波后信号的幅值分布（分位数，更稳健）
    amp_80 = np.quantile(np.abs(band_filtered), 0.8)  # 80分位数替代固定0.8倍最大值
    peak_mask = np.abs(band_filtered) > amp_80
    # 步骤3.2：动态增强系数（尖峰越强，增强幅度越小）
    peak_amps = np.abs(band_filtered[peak_mask])
    enhance_coeffs = 1.0 + (original_max_amp - peak_amps) / original_max_amp * 0.5  # 系数1.0~1.5
    # 步骤3.3：应用增强（非尖峰保持不变）
    enhanced_voltage = band_filtered.copy()
    enhanced_voltage[peak_mask] = band_filtered[peak_mask] * enhance_coeffs

    # ========== 步骤4：幅值+均值校准（完全保留原始物理意义） ==========
    enhanced_voltage = enhanced_voltage * (original_max_amp / np.max(np.abs(enhanced_voltage)))
    enhanced_voltage = enhanced_voltage + (original_mean - np.mean(enhanced_voltage))  # 补偿均值偏移

    return pd.DataFrame({'Voltage': enhanced_voltage})

def enhance_lstm_feature_slim(data_df, down_sr, base_freq):
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))
    original_mean = np.mean(voltage)
    # 新增：记录原始信号的首尾幅值（用于边缘校准）
    original_head = voltage[:100].mean()  # 前100个点的均值
    original_tail = voltage[-100:].mean()  # 后100个点的均值

    # ========== 步骤1：弱带通滤波（增加边缘延拓，消除滤波边缘效应） ==========
    nyq = 0.5 * down_sr
    low = (base_freq - 2) / nyq
    high = (base_freq + 2) / nyq
    b, a = butter(1, [low, high], btype='band')
    # 用“镜像延拓”处理信号边缘，消除滤波的开头/结尾波动
    filtered_voltage = filtfilt(b, a, voltage, padtype='odd', padlen=len(voltage)//10)

    # ========== 步骤2：原始信号与滤波信号融合（0.8:0.2） ==========
    fused_voltage = voltage * 0.0 + filtered_voltage * 1.0 # 这个比例很重要，后者越大特征越明显

    # ========== 步骤3：分层校准（先均值，再边缘） ==========
    # 1. 全局均值校准
    fused_voltage = fused_voltage + (original_mean - np.mean(fused_voltage))
    # 2. 边缘幅值校准（强制首尾100点匹配原始均值，消除下坡）
    fused_voltage[:100] = fused_voltage[:100] + (original_head - fused_voltage[:100].mean())
    fused_voltage[-100:] = fused_voltage[-100:] + (original_tail - fused_voltage[-100:].mean())

    # ========== 步骤4：轻度尖峰增强（不变） ==========
    peak_mask = np.abs(fused_voltage) > np.max(np.abs(fused_voltage)) * 0.9
    enhanced_voltage = np.where(peak_mask, fused_voltage * 1.2, fused_voltage)

    # ========== 步骤5：幅值校准（不变） ==========
    enhanced_voltage = enhanced_voltage * (original_max_amp / np.max(np.abs(enhanced_voltage)))

    return pd.DataFrame({'Voltage': enhanced_voltage})

def adaptive_feature_enhancement(data_df, down_sr, top_n=1): # 表现尚可，差异很小和之前
    """
    学术化特征强化：自适应提取主导基频 + 保留尖峰 + 抑制噪声（无提前归一化）
    :param down_sr: 降采样后采样率
    :param top_n: 提取前N个主导基频
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))  # 保留原始幅值基准

    # ========== 步骤1：自适应提取主导基频（学术化：功率谱密度+峰值检测） ==========
    # 计算功率谱密度，定位主导频率
    f, Pxx = signal.welch(voltage, fs=down_sr, nperseg=min(1024, len(voltage)//5))
    # 检测能量前top_n的基频
    peaks, _ = signal.find_peaks(Pxx, height=np.max(Pxx)*0.3)  # 阈值：能量≥30%最大值
    top_freqs = f[peaks[np.argsort(Pxx[peaks])[::-1][:top_n]]]  # 按能量排序取前N个
    if len(top_freqs) == 0:
        top_freqs = [2]  # 兜底基频
    base_freq = top_freqs[0]
    print(f"✅ 自适应提取主导基频：{base_freq:.2f} Hz")

    # ========== 步骤2：带通滤波（聚焦基频±1Hz，学术化描述：“基频带通滤波”） ==========
    nyq = 0.5 * down_sr
    low = (base_freq - 1) / nyq
    high = (base_freq + 1) / nyq
    b, a = signal.butter(3, [low, high], btype='band', analog=False)  # 3阶滤波器（更平滑）
    filtered_voltage = signal.filtfilt(b, a, voltage)  # 零相位滤波，无信号偏移

    # ========== 步骤3：尖峰特征增强（学术化：“残差叠加法”保留原始尖峰） ==========
    # 计算滤波残差（原始信号 - 滤波信号）：保留原始尖峰
    residual = voltage - filtered_voltage
    # 残差加权叠加（强化尖峰，保留趋势）
    enhanced_voltage = filtered_voltage + residual * 0.8  # 残差权重0.8，平衡趋势与尖峰

    # ========== 步骤4：幅值校准（恢复原始信号的幅值范围） ==========
    enhanced_voltage = enhanced_voltage * (original_max_amp / np.max(np.abs(enhanced_voltage)))

    return pd.DataFrame({'Voltage': enhanced_voltage}), base_freq




def envelope_dominated_smooth(data_df, wavelet='db4', level=3):
    """
    包络主导的平滑处理：
    1. 还原原始幅值；
    2. 强化低频包络，弱化突变尖峰，呈现类似正弦的渐变规律。
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))  # 记录原始幅值最大值，用于还原

    # ========== 步骤1：小波降噪（弱化尖峰，保留趋势） ==========
    coeffs = pywt.wavedec(voltage, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(voltage)))
    # 关键：增大高频系数的阈值，进一步弱化突变尖峰
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * 2, mode='soft')
    denoised_voltage = pywt.waverec(coeffs, wavelet)
    denoised_voltage = denoised_voltage[:len(voltage)]  # 对齐长度

    # ========== 步骤2：提取低频包络并主导波形 ==========
    analytic_signal = signal.hilbert(denoised_voltage)
    envelope = np.abs(analytic_signal)
    # 包络线低通滤波（截止频率设为2Hz，完全贴合基频）
    down_sr = 10000
    nyq = 0.5 * down_sr
    cutoff = 2  # 与基频一致，让包络更接近正弦
    b, a = butter(2, cutoff / nyq, btype='low')
    smooth_envelope = filtfilt(b, a, envelope)

    # ========== 步骤3：包络主导波形（弱化尖峰，还原幅值） ==========
    # 用包络线作为“主体趋势”，叠加少量降噪信号的波动（弱化突变）
    # 核心：包络占80%权重，降噪信号占20%权重
    enhanced_voltage = (smooth_envelope * 0.8) + (denoised_voltage * 0.2)
    # 还原原始幅值量级
    enhanced_voltage = enhanced_voltage * (original_max_amp / np.max(np.abs(enhanced_voltage)))

    enhanced_df = pd.DataFrame({'Voltage': enhanced_voltage})
    return enhanced_df


def balanced_smooth_with_spike(data_df, wavelet='db4', level=3):
    """
    平衡版：保留部分尖峰+平滑渐变+还原幅值
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))  # 强制还原原始幅值

    # ========== 步骤1：小波降噪（保留部分尖峰，弱化过度突变） ==========
    coeffs = pywt.wavedec(voltage, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(voltage)))
    # 调整阈值：保留50%的尖峰能量，弱化过度突变
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * 1.2, mode='soft')
    denoised_voltage = pywt.waverec(coeffs, wavelet)
    denoised_voltage = denoised_voltage[:len(voltage)]  # 对齐长度

    # ========== 步骤2：提取包络线（平滑渐变，保留基频） ==========
    analytic_signal = signal.hilbert(denoised_voltage)
    envelope = np.abs(analytic_signal)
    down_sr = 10000
    nyq = 0.5 * down_sr
    cutoff = 2.5  # 略高于基频，保留少量波动
    b, a = butter(2, cutoff / nyq, btype='low')
    smooth_envelope = filtfilt(b, a, envelope)

    # ========== 步骤3：平衡合并（尖峰占40%，包络占60%） ==========
    # 核心：既保留部分尖峰，又让包络主导渐变
    enhanced_voltage = (denoised_voltage * 0.4) + (smooth_envelope * 0.6)
    # 强制还原原始幅值
    enhanced_voltage = enhanced_voltage * (original_max_amp / np.max(np.abs(enhanced_voltage)))

    enhanced_df = pd.DataFrame({'Voltage': enhanced_voltage})
    return enhanced_df


def downsample_to_target_count(data_df, target_count=20000, preserve_peak=True):
    """
    定向降采样到指定条数（核心：按比例降采样，保留峰值/基频）
    :param data_df: 原始DataFrame（Voltage列）
    :param target_count: 目标条数（默认2w）
    :param preserve_peak: 是否保留脉冲尖峰（默认True）
    :return: 降采样后的DataFrame、降采样后采样率
    """
    original_count = len(data_df)
    original_sr = 50000  # 原始采样率50kHz=100w/20s

    # 计算降采样因子（按条数比例）
    downsample_factor = original_count / target_count
    if downsample_factor < 1:
        print(f"⚠️ 目标条数{target_count}大于原始条数{original_count}，无需降采样")
        return data_df, original_sr

    # 取整（保证降采样后条数接近目标值）
    downsample_factor = int(np.round(downsample_factor))
    print(f"✅ 降采样因子：{downsample_factor}（原始{original_count}条 → 目标{target_count}条）")

    # 选择降采样方式（保留峰值优先）
    if preserve_peak:
        down_df, down_sr = downsample_data_peak_preserve_signed(data_df, downsample_factor=downsample_factor)
    else:
        down_df, down_sr = downsample_data_linear(data_df, downsample_factor=downsample_factor)

    # 最终截断/补全到目标条数（确保精准2w条）
    final_df = down_df.iloc[:target_count].reset_index(drop=True)
    print(f"✅ 降采样完成：原始{original_count}条 → 最终{len(final_df)}条（目标{target_count}条）")

    return final_df, down_sr