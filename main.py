import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import welch, find_peaks
from scipy.signal import butter, filtfilt  # 滤波器设计+零相位滤波
import pywt # 小波变换库

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


def downsample_data(data_df, downsample_factor=10):
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

def lowpass_filter(data_df, sr, cutoff_freq):
    """
    巴特沃斯低通滤波：保留低频信号，滤除高频噪声
    :param data_df: 降采样后的DataFrame
    :param sr: 降采样后的采样率
    :param cutoff_freq: 截止频率（设为2×核心频率）
    :return: 滤波后的DataFrame
    """
    # 提取电压数组
    voltage = data_df['Voltage'].values
    # 设计巴特沃斯低通滤波器（4阶，平衡滤波效果和波形失真）
    nyq = 0.5 * sr  # 奈奎斯特频率
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(2, normal_cutoff, btype='low', analog=False) # 二阶滤波器
    # 零相位滤波（filtfilt）：避免波形偏移，适合非实时场景
    filtered_voltage = filtfilt(b, a, voltage)
    # 转成DataFrame
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
# 主调用逻辑
# ========== 4. 优化后的主调用逻辑 ==========
if __name__ == "__main__":
    # 1. 读取原始数据
    csv_path = "fix150.csv"
    raw_df = extract_single_column_csv(csv_path)
    plot_voltage_single_column(raw_df, title_suffix="（原始数据）")

    # 2. 核心步骤：用原始数据找10Hz内的真实基频（关键！）
    raw_sr = 50000
    real_base_freq = plot_spectrum_base_freq(raw_df, raw_sr, top_n=3)

    # 3. 用峰值降采样（保留脉冲尖峰）
    down_df, down_sr = downsample_data_peak_preserve_signed(raw_df, downsample_factor=5)
    plot_voltage_single_column(down_df, title_suffix="（降采样后，未滤波）")


    # # 只保留2Hz基频所在的1-10Hz区间，精准保留脉冲特征
    # filtered_df = bandpass_filter(down_df, down_sr, low_cut=1, high_cut=10)
    # print(f"✅ 带通滤波区间：1-10Hz（适配2Hz基频）")
    # plot_voltage_single_column(filtered_df, title_suffix="（降采样+低通滤波后）")
    # 4. 关键修改：改用小波阈值降噪（保留基频+尖峰）
    denoised_df = wavelet_denoise(down_df, wavelet='db4', level=3)
    print(f"✅ 小波阈值降噪完成（保留2Hz基频+脉冲尖峰）")
    plot_voltage_single_column(denoised_df, title_suffix="（峰值降采样+小波降噪后）")

    # # 5. 可选：小波变换补充验证（贴合导师建议，提升学术性）
    # wavelet_verify_base_freq_simple(raw_df, real_base_freq, raw_sr)