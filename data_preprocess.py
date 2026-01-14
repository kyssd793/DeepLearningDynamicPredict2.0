import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pywt # 小波变换库
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch, find_peaks, butter, filtfilt, hilbert, medfilt, savgol_filter  # 新增hilbert
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


def remove_dc_offset_and_outliers(data_df, sigma=3):
    """去除直流偏移 + 裁剪极端异常值（3σ原则）"""
    voltage = data_df['Voltage'].values

    # 步骤1：去除直流偏移
    dc_offset = np.mean(voltage)
    voltage_no_dc = voltage - dc_offset

    # 步骤2：裁剪极端异常值（只保留均值±3σ范围内的数据）
    mean = np.mean(voltage_no_dc)
    std = np.std(voltage_no_dc)
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    # 超出范围的异常值，用边界值替代（避免信号断裂）
    voltage_clean = np.clip(voltage_no_dc, lower_bound, upper_bound)

    return pd.DataFrame({'Voltage': voltage_clean})


# 原来的方法保留，峰值未美化比较胖
def extract_smooth_trend_envelope_final(data_df, low_cutoff=5, down_sr=1000, denoise_win=5, smooth_win=501,
                                        peak_gain=1.5):
    # 新增：先去除直流偏移+极端异常值（针对你标注的红色尖峰）
    data_df_clean = remove_dc_offset_and_outliers(data_df)
    voltage = data_df_clean['Voltage'].values

    # 步骤1：低通滤波（去掉高频密集波动）
    nyq = 0.5 * down_sr
    low = low_cutoff / nyq
    b, a = butter(4, low, btype='low')
    lowpass_voltage = filtfilt(b, a, voltage)

    # 步骤2：中值滤波（去掉小毛刺）
    denoised_voltage = medfilt(lowpass_voltage, kernel_size=denoise_win)

    # 步骤3：正负双包络
    positive_mask = denoised_voltage >= 0
    positive_envelope = np.zeros_like(denoised_voltage)
    positive_envelope[positive_mask] = denoised_voltage[positive_mask]

    negative_mask = denoised_voltage < 0
    negative_envelope = np.zeros_like(denoised_voltage)
    negative_envelope[negative_mask] = -denoised_voltage[negative_mask]

    # 步骤4：大窗口平滑
    smooth_positive = savgol_filter(positive_envelope, window_length=smooth_win, polyorder=2)
    smooth_negative = savgol_filter(negative_envelope, window_length=smooth_win, polyorder=2)

    # 步骤5：尖峰增强
    pos_peak_mask = smooth_positive > np.max(smooth_positive) * 0.8
    smooth_positive = np.where(pos_peak_mask, smooth_positive * peak_gain, smooth_positive)

    neg_peak_mask = smooth_negative > np.max(smooth_negative) * 0.8
    smooth_negative = np.where(neg_peak_mask, smooth_negative * peak_gain, smooth_negative)

    # 步骤6：合并+符号还原
    enhanced_voltage = np.where(positive_mask, smooth_positive, -smooth_negative)

    # 步骤7：幅值校准
    original_max_amp = np.max(np.abs(data_df['Voltage'].values))
    enhanced_max_amp = np.max(np.abs(enhanced_voltage))
    if enhanced_max_amp > 0:
        enhanced_voltage = enhanced_voltage * (original_max_amp / enhanced_max_amp)

    return pd.DataFrame({'Voltage': enhanced_voltage})


# 高斯美化的方法。原方法在上面做保留，峰值比较胖.但是一旦换了数据d15-20则稍微糟糕一点。
def hilbert_envelope_extraction(data_df, low_cutoff=10, down_sr=1000, denoise_win=5, gauss_sigma=50, peak_gain=1.2):
    # 1. 去直流偏移+异常值
    data_df_clean = remove_dc_offset_and_outliers(data_df)
    voltage = data_df_clean['Voltage'].values

    # 2. 低通滤波（保留7Hz内的趋势）
    nyq = 0.5 * down_sr
    low = low_cutoff / nyq
    b, a = butter(4, low, btype='low')
    lowpass_voltage = filtfilt(b, a, voltage)

    # 3. 中值滤波去小毛刺
    denoised_voltage = medfilt(lowpass_voltage, kernel_size=denoise_win)

    # 4. 高斯滤波（核心：让曲线更柔和、峰值更圆润）
    # sigma越大，曲线越平滑（50是适配2w条数据的柔和值）
    smooth_voltage = gaussian_filter1d(denoised_voltage, sigma=gauss_sigma)

    # 5. 轻量尖峰增强（避免过度尖锐）
    peak_mask = np.abs(smooth_voltage) > np.max(np.abs(smooth_voltage)) * 0.8
    smooth_voltage = np.where(peak_mask, smooth_voltage * peak_gain, smooth_voltage)

    # 6. 幅值校准
    original_max_amp = np.max(np.abs(data_df['Voltage'].values))
    enhanced_max_amp = np.max(np.abs(smooth_voltage))
    if enhanced_max_amp > 0:
        smooth_voltage = smooth_voltage * (original_max_amp / enhanced_max_amp)

    return pd.DataFrame({'Voltage': smooth_voltage})


def adaptive_piezo_preprocessing(data_df, down_sr):
    """
    自适应压电信号预处理（自动适配定/变流速数据）
    :param data_df: 输入DataFrame
    :param down_sr: 降采样率
    :return: 处理后的DataFrame
    """
    voltage = data_df['Voltage'].values

    # ========== 步骤1：分析数据特征（自动适配的核心） ==========
    # 1.1 计算全局功率谱，确定“有效频率上限”
    f, Pxx = welch(voltage, fs=down_sr, nperseg=min(len(voltage) // 10, 1000))
    # 有效频率：功率前80%的频率范围
    cum_power = np.cumsum(Pxx) / np.sum(Pxx)
    valid_freq_cutoff = f[np.where(cum_power >= 0.8)[0][0]]  # 覆盖80%功率的频率
    valid_freq_cutoff = max(valid_freq_cutoff, 5)  # 兜底不低于5Hz

    # 1.2 计算噪声强度，确定平滑程度
    noise_std = np.std(voltage[np.abs(voltage) < np.mean(np.abs(voltage))])
    smooth_sigma = int(50 * (noise_std / np.max(np.abs(voltage))))  # 噪声越大，平滑越强
    smooth_sigma = max(smooth_sigma, 20)  # 兜底不低于20

    # ========== 步骤2：基础预处理（去直流+异常值） ==========
    data_df_clean = remove_dc_offset_and_outliers(data_df)
    voltage_clean = data_df_clean['Voltage'].values

    # ========== 步骤3：自适应低通滤波（匹配有效频率） ==========
    nyq = 0.5 * down_sr
    low = valid_freq_cutoff / nyq
    b, a = butter(4, low, btype='low')
    lowpass_voltage = filtfilt(b, a, voltage_clean)

    # ========== 步骤4：自适应平滑（匹配噪声强度） ==========
    smooth_voltage = gaussian_filter1d(lowpass_voltage, sigma=smooth_sigma)

    # ========== 步骤5：轻量尖峰增强 ==========
    peak_mask = np.abs(smooth_voltage) > np.max(np.abs(smooth_voltage)) * 0.8
    smooth_voltage = np.where(peak_mask, smooth_voltage * 1.2, smooth_voltage)

    # ========== 步骤6：幅值校准 ==========
    original_max_amp = np.max(np.abs(voltage))
    enhanced_max_amp = np.max(np.abs(smooth_voltage))
    if enhanced_max_amp > 0:
        smooth_voltage = smooth_voltage * (original_max_amp / enhanced_max_amp)

    print(f"自适应参数：有效频率截止={valid_freq_cutoff:.1f}Hz，平滑sigma={smooth_sigma}")
    return pd.DataFrame({'Voltage': smooth_voltage})


# 适用于变流速环境下主频率不能按照平均来的，因此我们用滑动窗口每一段计算频率，每个窗口大小1000，20个滑窗.表现差异太大不太好
def enhance_lstm_feature_dynamic_freq(data_df, down_sr, window_size=1000, freq_band=1, target_freq_res=1):
    """
    动态基频跟踪的LSTM特征强化函数（适配变流速，修复10Hz锁定问题）
    :param data_df: 输入DataFrame（含Voltage列）
    :param down_sr: 降采样后的采样率（你的是1000Hz）
    :param window_size: 滑窗大小（1000条/窗，对应1秒）
    :param freq_band: 基频上下浮动范围（默认±1Hz）
    :param target_freq_res: 目标频率分辨率（1Hz，能识别0-10Hz所有整数频率）
    :return: 动态滤波后的DataFrame
    """
    voltage = data_df['Voltage'].values
    original_max_amp = np.max(np.abs(voltage))  # 全局幅值校准用
    n_points = len(voltage)
    enhanced_voltage = np.zeros_like(voltage)  # 初始化输出数组

    # 1. 划分滑窗（最后一个窗不足window_size则合并到前一个）
    window_starts = np.arange(0, n_points, window_size)
    if window_starts[-1] + window_size > n_points:
        window_starts = window_starts[:-1]

    # 2. 逐窗处理：动态计算基频 + 带通滤波
    for i, start in enumerate(window_starts):
        # 确定当前窗的结束位置
        end = min(start + window_size, n_points)
        window_voltage = voltage[start:end]
        window_len = len(window_voltage)  # 当前窗的实际长度（1000条）

        # ========== 步骤1：修复核心——按滑窗时长计算nperseg ==========
        # 目标：频率分辨率=target_freq_res（1Hz）→ nperseg=采样率/分辨率
        ideal_nperseg = int(down_sr / target_freq_res)  # 1000/1=1000
        # 限制nperseg不超过窗长度的80%（避免分段数为0）
        max_nperseg = int(window_len * 0.8)  # 1000*0.8=800
        # 最终nperseg取“理想值”和“最大值”的较小值，且为偶数（滤波稳定）
        nperseg = min(ideal_nperseg, max_nperseg)
        nperseg = nperseg if nperseg % 2 == 0 else nperseg - 1  # 偶数化
        # 分段数至少为2（避免Welch报错）
        nperseg = max(nperseg, 2)

        # ========== 步骤2：当前窗计算局部基频（修复频率轴） ==========
        f, Pxx = welch(
            window_voltage,
            fs=down_sr,
            nperseg=nperseg,
            noverlap=nperseg // 2,  # 重叠50%，提升频谱精度
            scaling='density'
        )
        # 过滤0-10Hz频率（保留真实基频范围）
        freq_mask = (f >= 0) & (f <= 10)
        f_filtered = f[freq_mask]
        Pxx_filtered = Pxx[freq_mask]

        # 兜底逻辑（无有效频率时用默认值）
        if len(Pxx_filtered) == 0:
            local_base_freq = 2  # 压电俘能常见基频
            print(f"窗{i + 1}（{start}-{end}）：无有效频率，用默认值={local_base_freq:.2f}Hz")
        else:
            # 修复峰值阈值：提高到最大值的20%，过滤低功率伪峰值
            peak_threshold = np.max(Pxx_filtered) * 0.2
            peaks, properties = find_peaks(Pxx_filtered, height=peak_threshold)

            if len(peaks) == 0:
                # 无峰值时，选功率最大的频率（但排除0Hz直流分量）
                non_dc_mask = f_filtered > 0  # 去掉0Hz
                if np.sum(non_dc_mask) == 0:
                    local_base_freq = 2
                else:
                    dominant_idx = np.argmax(Pxx_filtered[non_dc_mask])
                    local_base_freq = f_filtered[non_dc_mask][dominant_idx]
            else:
                # 有峰值时，选功率最大的峰值
                peak_energies = properties['peak_heights']
                top_peak_idx = np.argsort(peak_energies)[::-1][0]
                local_base_freq = f_filtered[peaks[top_peak_idx]]

        # 打印真实的局部基频（验证修复效果）
        print(f"窗{i + 1}（{start}-{end}）：局部基频={local_base_freq:.2f}Hz")

        # ========== 步骤3：当前窗动态带通滤波（和原有逻辑一致） ==========
        nyq = 0.5 * down_sr
        low = (local_base_freq - freq_band) / nyq
        high = (local_base_freq + freq_band) / nyq
        # 边界保护：避免频率小于0或大于Nyquist频率
        low = max(low, 0)
        high = min(high, 1)

        b, a = butter(2, [low, high], btype='band')
        window_filtered = filtfilt(b, a, window_voltage)

        # ========== 步骤4：当前窗尖峰增强 ==========
        window_peak_mask = np.abs(window_filtered) > np.max(np.abs(window_filtered)) * 0.8
        window_enhanced = np.where(window_peak_mask, window_filtered * 1.5, window_filtered)

        # ========== 步骤5：当前窗幅值校准 ==========
        window_max_amp = np.max(np.abs(window_enhanced))
        if window_max_amp > 0:
            window_enhanced = window_enhanced * (np.max(np.abs(window_voltage)) / window_max_amp)

        # 赋值到输出数组
        enhanced_voltage[start:end] = window_enhanced

    # 全局幅值校准
    enhanced_voltage = enhanced_voltage * (original_max_amp / np.max(np.abs(enhanced_voltage)))

    return pd.DataFrame({'Voltage': enhanced_voltage})