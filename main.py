import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

# 适配单列数据的绘图函数
def plot_voltage_single_column(data):
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
    plt.title("电压随数据序列变化（100w条数据，抽样率1/{}）".format(sample_rate))
    plt.grid(alpha=0.3)  # 增加网格，便于查看
    plt.show()

# 主调用逻辑
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_path = "fix150.csv"
    # 读取数据
    voltage_df = extract_single_column_csv(csv_path)
    # 绘制图形
    plot_voltage_single_column(voltage_df)