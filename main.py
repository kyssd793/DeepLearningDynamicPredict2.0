import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import welch, find_peaks
from scipy.signal import butter, filtfilt  # 新增：滤波器设计+零相位滤波

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

# 主调用逻辑
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_path = "fix150.csv"
    # 读取数据
    voltage_df = extract_single_column_csv(csv_path)
    plot_voltage_single_column(voltage_df,title_suffix="原始图")
    # 降采样，降采样因子是使得采样率的除数
    down_df, down_sr = downsample_data_peak_preserve_signed(voltage_df, downsample_factor=5)
    # 频谱分析，找压电片主频率
    main_freq = plot_spectrum(down_df, down_sr, top_n=3)
    # 低通滤波，截止频率设置为主频率的5倍
    filtered_df=lowpass_filter(down_df, down_sr, cutoff_freq=500)
    # 对比滤波前后的波形
    plot_voltage_single_column(down_df, title_suffix="（降采样后，未滤波）")
    plot_voltage_single_column(filtered_df, title_suffix="（降采样+低通滤波后）")