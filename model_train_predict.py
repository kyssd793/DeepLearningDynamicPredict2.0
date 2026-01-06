import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Attention
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.keras.saving.save import load_model
import pickle


def save_to_csv(time_data, predicted_data_inversed, filename):
    # 如果是二维数组或其他类型，可以展平为一维数组
    time_data = time_data.flatten() if hasattr(time_data, 'flatten') else time_data
    predicted_data_inversed = predicted_data_inversed.flatten() if hasattr(predicted_data_inversed, 'flatten') else predicted_data_inversed

    # 创建 DataFrame
    df = pd.DataFrame({
        'time': time_data,
        'predicted': predicted_data_inversed
    })

    # 保存为 CSV 文件
    df.to_csv(filename, index=False)


def save_comparison_plot(true_data, pred_data, time_true, time_pred,
                            filename='prediction_comparison.png', dpi=300):
    # 创建画布（适配大量数据的尺寸）
    fig, ax = plt.subplots(figsize=(15, 6), dpi=dpi)

    # 绘制真实数据（绿色）
    ax.plot(time_true, true_data['Voltage'].values,
            color='#2ecc71', label='True Data', alpha=0.8, linewidth=0.8)
    # 绘制预测数据（红色）
    ax.plot(time_pred, pred_data,
            color='#e74c3c', label='Predicted Data', alpha=0.8, linewidth=0.8)

    # 图表配置（英文标签，避免字体问题）
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title('True vs Predicted Voltage Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 紧凑布局，避免裁剪
    plt.tight_layout()
    # 保存图片（高清，无白边）
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # 关闭画布，释放内存

    print(f"✅ 对比图已保存为PNG：{os.path.abspath(filename)}")
    return os.path.abspath(filename)

def save_prediction_to_csv(pred_data, filename='prediction_result.csv'):
    # 展平数据为一维，确保单列
    pred_data_flat = pred_data.flatten()
    # 创建单列DataFrame
    df = pd.DataFrame({
        'Predicted_Voltage': pred_data_flat
    })
    # 保存CSV（无索引，仅数据）
    df.to_csv(filename, index=False, header=False)
    print(f"✅ 预测数据已保存为CSV：{os.path.abspath(filename)}")
    return os.path.abspath(filename)

def ensemble_loss(y_true, y_pred, a=0.3, b=0.7):
    # a=0.3 b=0.7 55开时候应该最好，强于37，强于64
    # 现在是64最好，且要每次不用模型预测
    # 保留64，在数据集20-18-60-32.5上效果很差
    # 定义集成式损失函数，根据需要调整权重 a 和 b
    loss1 = tf.square(y_true - y_pred)  # 第一个损失函数
    loss2 = tf.abs(y_true - y_pred)     # 第二个损失函数
    return a * loss1 + b * loss2        # 加权求和


# 新增：全局标准化函数（训练和预测用同一个scaler，避免分布偏移）
def create_scaler(data):
    """基于训练数据创建scaler，供后续预测使用"""
    scaler = MinMaxScaler()
    scaler.fit(data[['Voltage']])
    return scaler


# ========== 3. 构建数据集（不变） ==========
def create_sequences(data_scaled, seq_length=32):
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i + seq_length, 0])
        y.append(data_scaled[i + seq_length, 0])

    if len(X) == 0:
        raise ValueError(f"数据长度不足！至少需要 {seq_length + 1} 个样本（当前：{len(data_scaled)}）")

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


def train_lstm_attention_model(preprocessed_df, seq_length=32, save_path='./',
                               scaler=None, roll_window_ratio=0.2, roll_step_ratio=0.1):
    """
    带滚动验证的LSTM-Attention训练（适配时序数据，避免分布偏移）
    :param preprocessed_df: 预处理后的DataFrame（Voltage列）
    :param seq_length: 时间步
    :param save_path: 模型保存路径
    :param scaler: 外部scaler
    :param roll_window_ratio: 每个验证窗口占总数据的比例（默认20%）
    :param roll_step_ratio: 滚动步长占总数据的比例（默认10%）
    :return: 训练好的模型、scaler
    """
    # 1. 标准化
    if scaler is None:
        scaler = create_scaler(preprocessed_df)
    data_scaled = scaler.transform(preprocessed_df[['Voltage']])

    # 2. 构建完整序列数据集
    X_all, y_all = create_sequences(data_scaled, seq_length)
    total_samples = len(X_all)

    # 3. 滚动验证参数计算
    val_window_size = int(total_samples * roll_window_ratio)  # 每个验证窗口大小（20%）
    roll_step = int(total_samples * roll_step_ratio)  # 滚动步长（10%）
    if val_window_size == 0 or roll_step == 0:
        raise ValueError("数据量过小，无法进行滚动验证！请增大数据量或调整窗口/步长比例")

    # 4. 模型结构（和你原代码一致）
    def build_model():
        inputs = Input(shape=(seq_length, 1))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv2)
        lstm1 = LSTM(units=50, return_sequences=True)(pool1)
        attention_out = Attention()([lstm1, lstm1])
        lstm2 = LSTM(units=50, return_sequences=False)(attention_out)
        lstm2 = Dropout(0.1)(lstm2)
        outputs = Dense(1)(lstm2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=ensemble_loss)
        return model

    # 5. 模型路径
    # 5. 模型路径（修改：保存权重而非完整模型）
    model_weights_path = os.path.join(save_path, 'lstm_model_weights.h5')  # 权重文件
    scaler_filename = 'scaler_piezo.pkl'
    scaler_path = os.path.join(save_path, scaler_filename)

    # 6. 加载/初始化模型
    model = build_model()  # 先重建模型结构
    # 新增：如果权重文件存在，直接加载，不训练
    if os.path.exists(model_weights_path):
        print(f"✅ 发现已有权重文件，直接加载：{model_weights_path}")
        model.load_weights(model_weights_path)
        # 同时加载scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        print(f"✅ 跳过训练，直接使用已有模型")
        return model, scaler
    # 否则，开始训练
    else:
        print("初始化新模型，开始滚动训练...")

    # 7. 滚动验证训练（核心逻辑）
    start_idx = 0
    val_loss_list = []  # 记录每个窗口的验证损失
    while start_idx + val_window_size <= total_samples:
        # 划分当前训练/验证窗口（严格按时间顺序，不打乱）
        train_end_idx = start_idx
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_window_size

        X_train = X_all[:train_end_idx] if train_end_idx > 0 else X_all[:val_start_idx]
        y_train = y_all[:train_end_idx] if train_end_idx > 0 else y_all[:val_start_idx]
        X_val = X_all[val_start_idx:val_end_idx]
        y_val = y_all[val_start_idx:val_end_idx]

        # 跳过样本不足的情况
        if len(X_train) < 100 or len(X_val) < 50:  # 最小样本阈值
            start_idx += roll_step
            continue

        # 训练当前窗口（增量训练，贴合时序分布）
        print(f"\n=== 滚动窗口 {start_idx // roll_step + 1} ===")
        print(f"训练集：0 ~ {train_end_idx if train_end_idx > 0 else val_start_idx} 样本")
        print(f"验证集：{val_start_idx} ~ {val_end_idx} 样本")

        history = model.fit(
            X_train, y_train,
            epochs=2,  # 每个窗口少量epochs，避免过拟合
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1,
            shuffle=False  # 时序数据绝对不能shuffle！
        )

        # 记录验证损失
        val_loss = history.history['val_loss'][-1]
        val_loss_list.append(val_loss)
        print(f"当前窗口验证损失：{val_loss:.4f}")

        # 滚动到下一个窗口
        start_idx += roll_step

    # 8. 保存最终模型和scaler
    if not os.path.exists(model_weights_path):
        model.save_weights(model_weights_path)  # 保存权重
        print(f"\n滚动训练完成，模型权重保存至：{model_weights_path}")
    print(f"平均验证损失：{np.mean(val_loss_list):.4f}")

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler保存至：{scaler_path}")

    return model, scaler


def predict_with_sliding_window_fixed(dataB, seq_length=32, model_weights_path='./lstm_model_weights.h5',
                                      scaler_path='./scaler_piezo.pkl', future_steps=16, target_total_points=None):
    """
    修正后的预测逻辑：对齐代码1，分段预测+真实数据重置窗口（适配无Time(s)列，用采样点序号作为时间）
    :param dataB: 输入数据（DataFrame，仅含Voltage列）
    :param seq_length: 时间步长（窗口大小）
    :param model_path: 模型路径（适配咱们的模型命名）
    :param scaler_path: 标准化器路径（适配咱们的scaler命名）
    :param future_steps: 单次预测长度（每段预测的点数，保持16不变）
    :param target_total_points: 目标总预测点数（可选，默认预测到数据B的最大长度）
    :return: 预测采样点序号（替代时间）、逆标准化后的预测电压值
    """
    import pickle
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 2. 重建模型结构 + 加载权重（核心）
    def build_model():  # 复制训练时的模型结构
        inputs = Input(shape=(seq_length, 1))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv2)
        lstm1 = LSTM(units=50, return_sequences=True)(pool1)
        attention_out = Attention()([lstm1, lstm1])
        lstm2 = LSTM(units=50, return_sequences=False)(attention_out)
        lstm2 = Dropout(0.1)(lstm2)
        outputs = Dense(1)(lstm2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=ensemble_loss)
        return model

    model = build_model()
    model.load_weights(model_weights_path)  # 加载权重
    print(f"✅ 成功加载模型权重：{model_weights_path}")

    # 适配列名：仅保留Voltage（咱们的数据列）
    col_name = 'Voltage' if 'Voltage' in dataB.columns else 'CH1V'

    # 标准化
    dataB_scaled = scaler.transform(dataB[[col_name]])

    # 构建测试集窗口（完全保留你的逻辑）
    X_test = []
    for i in range(len(dataB_scaled) - seq_length):
        X_test.append(dataB_scaled[i:i + seq_length, 0])
    X_test = np.array(X_test).reshape(-1, seq_length, 1)

    # 动态计算总段数（保留你的核心优化）
    max_possible_steps = len(X_test)
    if target_total_points is not None:
        total_steps = int(np.ceil(target_total_points / future_steps))
    else:
        total_steps = max_possible_steps
    total_steps = min(total_steps, max_possible_steps)
    if total_steps == 0:
        raise ValueError(f"数据B长度不足！X_test仅{len(X_test)}个样本，无法完成至少1段预测")

    # 加载模型
    model = tf.keras.models.load_model(model_weights_path, custom_objects={'ensemble_loss': ensemble_loss})

    all_predicted_data = []
    time_list = []  # 现在存储的是采样点序号，替代Time(s)
    for step in range(total_steps):
        start_index = step * future_steps
        if start_index >= len(X_test):
            break
        last_sequence = X_test[start_index].reshape(1, seq_length, 1)

        # 单次预测future_steps个点（完全保留你的逻辑）
        predicted_data = []
        for i in range(future_steps):
            predicted_value = model.predict(last_sequence, verbose=0)
            predicted_data.append(predicted_value[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = predicted_value[0, 0]

        # ========== 核心适配：用采样点序号替代Time(s) ==========
        all_predicted_data.extend(predicted_data)
        # 起始采样点：start_index + seq_length（和你原逻辑对齐）
        start_point = start_index + seq_length
        # 生成采样点序号（替代时间轴）：[start_point, start_point+1, ..., start_point+future_steps-1]
        time_list.extend([start_point + i for i in range(len(predicted_data))])

    # 逆标准化
    all_predicted_data = np.array(all_predicted_data).reshape(-1, 1)
    all_predicted_data_inversed = scaler.inverse_transform(all_predicted_data)
    time_data = np.array(time_list)  # 现在是采样点序号，不是时间

    print(f"✅ 预测完成：共分{total_steps}段，单次预测{future_steps}点，总预测{len(all_predicted_data)}点")
    return time_data, all_predicted_data_inversed


# 以固定窗口，32-》生成32-64的生成值，32-64真实值生成64-96的生成值，以此类推
def predict_stepped_window_fast(dataB, seq_length=32, model_weights_path='./lstm_model_weights.h5',
                                scaler_path='./scaler_piezo.pkl', predict_step=32, target_total_points=None):
    """
    步进式窗口预测（按你的思路优化：牺牲少量连滚精度，换取极致速度）
    :param dataB: 输入数据（DataFrame，仅含Voltage列）
    :param seq_length: 输入窗口长度（固定32）
    :param model_path: 模型路径
    :param scaler_path: 标准化器路径
    :param predict_step: 单次预测点数（和输入窗口等长，固定32）
    :param target_total_points: 目标总预测点数（默认20万）
    :return: 预测采样点序号、逆标准化后的预测电压值
    """
    import pickle
    import numpy as np
    from tensorflow.keras.models import load_model
    import tensorflow as tf


    # 1. 加载scaler和模型
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 2. 重建模型结构 + 加载权重
    def build_model():  # 复制训练时的模型结构
        inputs = Input(shape=(seq_length, 1))
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv2)
        lstm1 = LSTM(units=50, return_sequences=True)(pool1)
        attention_out = Attention()([lstm1, lstm1])
        lstm2 = LSTM(units=50, return_sequences=False)(attention_out)
        lstm2 = Dropout(0.1)(lstm2)
        outputs = Dense(1)(lstm2)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=ensemble_loss)
        return model

    model = build_model()
    model.load_weights(model_weights_path)
    print(f"✅ 成功加载模型权重：{model_weights_path}")

    # 2. 适配列名+标准化
    col_name = 'Voltage' if 'Voltage' in dataB.columns else 'CH1V'
    data_scaled = scaler.transform(dataB[[col_name]])
    data_length = len(data_scaled)

    # 3. 确定目标预测量（默认20万）
    if target_total_points is None:
        target_total_points = data_length
    # 确保预测点数是predict_step的整数倍
    target_total_points = ((target_total_points + predict_step - 1) // predict_step) * predict_step

    # 4. 初始化变量
    all_pred_time = []  # 预测点序号
    all_pred_data = []  # 预测值（标准化）
    current_start = 0  # 当前真实数据起始位置

    # 5. 核心：步进式预测（每轮用32真实数据→预测32点→步进32）
    while len(all_pred_data) < target_total_points:
        # 边界校验：确保真实数据足够取32个
        if current_start + seq_length > data_length:
            break

        # 取当前轮的真实数据窗口（核心：用真实数据重置，避免误差累积）
        current_window = data_scaled[current_start:current_start + seq_length, 0]
        current_window = current_window.reshape(1, seq_length, 1)  # 模型输入格式

        # 单次预测32个点（替代原来的逐点预测）
        predicted_data = []
        temp_window = current_window.copy()
        for i in range(predict_step):
            # 预测1个点
            pred = model.predict(temp_window, verbose=0)[0, 0]
            predicted_data.append(pred)
            # 窗口滚动（用预测值填充，仅本轮内滚动）
            temp_window = np.roll(temp_window, -1, axis=1)
            temp_window[0, -1, 0] = pred

        # 计算当前轮预测点的序号
        pred_start = current_start + seq_length  # 预测点从真实窗口后开始
        pred_time = [pred_start + i for i in range(predict_step)]

        # 累加结果
        all_pred_data.extend(predicted_data)
        all_pred_time.extend(pred_time)

        # 步进：下一轮取新的32个真实数据（核心：用真实数据重置）
        current_start += predict_step

        # 打印进度（方便监控）
        progress = min(len(all_pred_data) / target_total_points * 100, 100)
        print(f"预测进度：{progress:.1f}% | 已预测：{len(all_pred_data)}/{target_total_points} 点", end='\r')

    # 截断到目标点数（避免超出）
    all_pred_data = all_pred_data[:target_total_points]
    all_pred_time = all_pred_time[:target_total_points]

    # 逆标准化
    pred_data_scaled = np.array(all_pred_data).reshape(-1, 1)
    pred_data_inversed = scaler.inverse_transform(pred_data_scaled)
    pred_time = np.array(all_pred_time)

    print(f"\n✅ 步进式预测完成：总预测{len(pred_data_inversed)}点，单次窗口{seq_length}，单次预测{predict_step}点")
    return pred_time, pred_data_inversed

def plot_predicted_data(time_data, predicted_data_inversed):
    # 绘制预测结果
    plt.plot(time_data, predicted_data_inversed, label="Predicted Voltage")
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    # plt.title("预测的电压随时间变化")
    plt.legend()
    plt.show()


# ========== 新增：你需要的双图对比绘制函数 ==========
def plot_double_figure(true_data, time_true, pred_data, time_pred):
    """
    绘制两张对比图（x轴为采样点序号，替代时间）：
    图1：真实数据 + 预测数据对比
    图2：仅预测数据独立展示
    """
    # 图1：真实电压 vs 预测电压（x轴为采样点序号）
    plt.figure(figsize=(12, 5))
    plt.plot(time_true, true_data['Voltage'].values, label='真实电压', alpha=0.7, color='blue')
    plt.plot(time_pred, pred_data, label='预测电压', alpha=0.7, color='red')
    plt.xlabel('采样点序号')  # 替换为采样点
    plt.ylabel('电压 (V)')
    plt.title('真实电压 vs 预测电压对比（采样点序号）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 图2：仅预测电压（x轴为采样点序号）
    plt.figure(figsize=(12, 5))
    plt.plot(time_pred, pred_data, color='red', label='预测电压')
    plt.xlabel('采样点序号')  # 替换为采样点
    plt.ylabel('电压 (V)')
    plt.title('预测电压随采样点变化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


